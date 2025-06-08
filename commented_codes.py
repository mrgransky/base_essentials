@torch.no_grad()
def get_validation_metrics_old(
		model: torch.nn.Module,
		validation_loader: torch.utils.data.DataLoader,
		criterion: torch.nn.Module,
		device: torch.device,
		topK_values: List[int],
		cache_dir: str,
		finetune_strategy: str = None,
		chunk_size: int = 1024,
		verbose: bool = True,
		max_in_batch_samples: Optional[int] = None,
		force_recompute: bool = False,
		embeddings_cache: tuple = None,
		lora_params: Optional[Dict] = None,
		is_training: bool = False,
		model_hash: str = None,
) -> Dict:
		model.eval()
		torch.cuda.empty_cache()
		start_time = time.time()

		if finetune_strategy is None:
				finetune_strategy = "pretrained"

		model_class_name = model.__class__.__name__
		model_arch_name = getattr(model, 'name', 'unknown_arch')
		dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
		num_workers = getattr(validation_loader, 'num_workers', 'unknown_num_workers')

		try:
				class_names = validation_loader.dataset.dataset.classes
		except AttributeError:
				class_names = validation_loader.dataset.unique_labels

		n_classes = len(class_names)

		cache_file = os.path.join(
				cache_dir,
				f"{dataset_name}_{finetune_strategy}_bs_{validation_loader.batch_size}_nw_{num_workers}_{model_class_name}_{re.sub(r'[/@]', '_', model_arch_name)}_validation_embeddings.pt"
		)
		if model_hash:
				cache_file = cache_file.replace(".pt", f"_{model_hash}.pt")

		# Step 1: In-batch metrics (small subset)
		in_batch_metrics = None
		if max_in_batch_samples is not None:
				if verbose:
						print(f"Computing in-batch metrics with {max_in_batch_samples} samples...")
				in_batch_metrics = compute_direct_in_batch_metrics(
						model=model,
						validation_loader=validation_loader,
						criterion=criterion,
						device=device,
						topK_values=topK_values,
						max_samples=max_in_batch_samples
				)

		# Step 2: Load or compute embeddings
		cache_loaded = False
		if not is_training and embeddings_cache is not None:
				all_image_embeds, _ = embeddings_cache
				all_labels = torch.tensor(
						[validation_loader.dataset.labels_int[i] for i in range(len(validation_loader.dataset))],
						device='cpu'
				)
				cache_loaded = True
				if verbose:
						print("Loaded embeddings from provided cache.")
		elif not is_training and os.path.exists(cache_file) and not force_recompute:
				if verbose:
						print(f"Loading cached embeddings from {cache_file}")
				cached = torch.load(cache_file, map_location='cpu')
				all_image_embeds = cached['image_embeds']
				all_labels = cached['labels']
				cache_loaded = True

		if not cache_loaded or is_training:
			if verbose:
				print("Computing embeddings from scratch...")
			all_image_embeds = []
			all_labels = []
			model = model.to(device)
			model.eval()
			for images, _, labels_indices in tqdm(validation_loader, desc="Encoding images"):
				images = images.to(device, non_blocking=True)
				with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
					image_embeds = model.encode_image(images)
				image_embeds = F.normalize(image_embeds.float(), dim=-1)  # Ensure float32 after normalization
				all_image_embeds.append(image_embeds.cpu())
				all_labels.append(labels_indices.cpu())  # [batch_size] for single-label, [batch_size, num_classes] for multi-label
				# all_labels.extend(labels_indices.cpu().tolist())
			all_image_embeds = torch.cat(all_image_embeds, dim=0)
			all_labels = torch.cat(all_labels, dim=0)  # [num_samples] or [num_samples, num_classes]
			# all_labels = torch.tensor(all_labels, device='cpu')
			if not is_training:
				os.makedirs(cache_dir, exist_ok=True)
				torch.save({'image_embeds': all_image_embeds, 'labels': all_labels}, cache_file)
				if verbose:
					print(f"Saved embeddings to {cache_file}")

		# Step 3: Compute text embeddings
		text_inputs = clip.tokenize(class_names).to(device)
		with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
				class_text_embeds = model.encode_text(text_inputs)
		class_text_embeds = F.normalize(class_text_embeds.float(), dim=-1).cpu()  # Ensure float32 after normalization

		# Step 4: Compute similarity matrices
		device_image_embeds = all_image_embeds.to(device).float()
		device_class_text_embeds = class_text_embeds.to(device).float()
		device_labels = all_labels.to(device)

		i2t_similarity = device_image_embeds @ device_class_text_embeds.T
		t2i_similarity = device_class_text_embeds @ device_image_embeds.T

		# Step 5: Full-set metrics
		full_metrics = compute_full_set_metrics_from_cache(
				i2t_similarity=i2t_similarity,
				t2i_similarity=t2i_similarity,
				labels=device_labels,
				n_classes=n_classes,
				topK_values=topK_values,
				device=device
		)

		# Step 6: Retrieval metrics
		cache_key_base = f"{dataset_name}_{finetune_strategy}_{model_class_name}_{re.sub(r'[/@]', '_', model_arch_name)}"
		if lora_params:
				cache_key_base += f"_lora_rank_{lora_params['lora_rank']}_lora_alpha_{lora_params['lora_alpha']}_lora_dropout_{lora_params['lora_dropout']}"

		img2txt_metrics = compute_retrieval_metrics_from_similarity(
				similarity_matrix=i2t_similarity,
				query_labels=device_labels,
				candidate_labels=torch.arange(n_classes, device=device),
				topK_values=topK_values,
				mode="Image-to-Text",
				cache_dir=cache_dir,
				cache_key=f"{cache_key_base}_img2txt",
				is_training=is_training,
				verbose=verbose,
		)

		class_counts = torch.bincount(device_labels, minlength=n_classes)
		txt2img_metrics = compute_retrieval_metrics_from_similarity(
				similarity_matrix=t2i_similarity,
				query_labels=torch.arange(n_classes, device=device),
				candidate_labels=device_labels,
				topK_values=topK_values,
				mode="Text-to-Image",
				class_counts=class_counts,
				cache_dir=cache_dir,
				cache_key=f"{cache_key_base}_txt2img",
				is_training=is_training,
				verbose=verbose,
		)

		if verbose:
				print(f"Validation evaluation completed in {time.time() - start_time:.2f} sec")

		return {
				"in_batch_metrics": in_batch_metrics,
				"full_metrics": full_metrics,
				"img2txt_metrics": img2txt_metrics,
				"txt2img_metrics": txt2img_metrics
		}

def compute_full_set_metrics_from_cache_old(
		i2t_similarity: torch.Tensor,
		t2i_similarity: torch.Tensor,
		labels: torch.Tensor,
		n_classes: int,
		topK_values: List[int],
		device: str
	) -> Dict:
	# Filter valid K values
	valid_k_values = [k for k in topK_values if k <= n_classes]
	
	# Image-to-text accuracy metrics
	img2txt_preds = torch.argmax(i2t_similarity, dim=1)
	img2txt_acc = (img2txt_preds == labels).float().mean().item()
	
	# Image-to-text top-K accuracy
	img2txt_topk_acc = {}
	for k in valid_k_values:
			topk_indices = i2t_similarity.topk(k, dim=1)[1]
			correct = (topk_indices == labels.unsqueeze(1)).any(dim=1)
			img2txt_topk_acc[k] = correct.float().mean().item()
	
	# Text-to-image top-K accuracy
	txt2img_topk_acc = {}
	for k in topK_values:
			class_correct = 0
			effective_k = min(k, i2t_similarity.size(0))
			
			topk_indices = t2i_similarity.topk(effective_k, dim=1)[1]
			for class_idx in range(n_classes):
					retrieved_labels = labels[topk_indices[class_idx]]
					if class_idx in retrieved_labels:
							class_correct += 1
			
			txt2img_topk_acc[k] = class_correct / n_classes
	
	# Set top-1 text-to-image accuracy
	txt2img_acc = txt2img_topk_acc.get(1, 0.0)
	
	# Compute MRR (Mean Reciprocal Rank)
	ranks = i2t_similarity.argsort(dim=1, descending=True)
	rr_indices = ranks.eq(labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
	img2txt_mrr = (1.0 / rr_indices.float()).mean().item()
	
	# Compute cosine similarity (between corresponding image-text pairs)
	# We don't have direct pairs in this context, so using MRR as substitute
	
	# Return metrics in the expected format
	return {
		"img2txt_acc": float(img2txt_acc),
		"txt2img_acc": float(txt2img_acc),
		"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
		"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
		"mean_reciprocal_rank": float(img2txt_mrr),
		"cosine_similarity": 0.0  # Using placeholder since we don't have direct pairs
	}

def compute_retrieval_metrics_from_similarity_old(
		similarity_matrix: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor,
		topK_values: List[int],
		mode: str = "Image-to-Text",
		class_counts: Optional[torch.Tensor] = None,
		max_k: Optional[int] = None,
		cache_dir: str = None,
		cache_key: str = None,
		is_training: bool = False,
		verbose: bool = True,
) -> Dict:
		num_queries, num_candidates = similarity_matrix.shape
		device = similarity_matrix.device
		
		# Check cache only if not training
		cache_file = None
		if cache_dir and cache_key and not is_training:
				cache_file = os.path.join(cache_dir, f"{cache_key}_retrieval_metrics.json")
				if os.path.exists(cache_file):
						try:
								if verbose:
										print(f"Loading cached retrieval metrics from {cache_file}")
								with open(cache_file, 'r') as f:
										return json.load(f)
						except Exception as e:
								print(f"Error loading cache: {e}. Computing metrics.")
		
		if verbose:
				print(f"Computing retrieval metrics for {mode} (cache skipped: is_training={is_training})")
		
		if max_k is not None:
				valid_K_values = [K for K in topK_values if K <= max_k]
		else:
				valid_K_values = topK_values
		
		metrics = {"mP": {}, "mAP": {}, "Recall": {}}
		
		all_sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
		
		for K in valid_K_values:
				top_k_indices = all_sorted_indices[:, :K]
				retrieved_labels = candidate_labels[top_k_indices]
				true_labels_expanded = query_labels.unsqueeze(1).expand(-1, K)
				correct_mask = (retrieved_labels == true_labels_expanded)
				
				metrics["mP"][str(K)] = correct_mask.float().mean(dim=1).mean().item()
				
				if mode == "Image-to-Text":
						metrics["Recall"][str(K)] = correct_mask.any(dim=1).float().mean().item()
				else:
						relevant_counts = class_counts[query_labels]
						metrics["Recall"][str(K)] = (correct_mask.sum(dim=1) / relevant_counts.clamp(min=1)).mean().item()
				
				# Vectorized AP
				positions = torch.arange(1, K + 1, device=device).float().unsqueeze(0).expand(num_queries, K)
				cumulative_correct = correct_mask.float().cumsum(dim=1)
				precisions = cumulative_correct / positions
				ap = (precisions * correct_mask.float()).sum(dim=1) / correct_mask.sum(dim=1).clamp(min=1)
				metrics["mAP"][str(K)] = ap.nanmean().item()
		
		# Save to cache only if not training
		if cache_dir and cache_key and not is_training:
				try:
						os.makedirs(cache_dir, exist_ok=True)
						with open(cache_file, 'w') as f:
								json.dump(metrics, f)
						if verbose:
								print(f"Saved metrics to {cache_file}")
				except Exception as e:
						print(f"Warning: Failed to save cache {e}")
		
		return metrics



METADATA_PATTERNS = [
	r'bildetekst \w+',        # Matches 'bildetekst german' and similar
	r'kunststoff \w+',        # Matches photography material descriptions
	r'arkivreferanse \w+',    # Archive references
	r'\w+ pa \d+',            # Reference codes like 'ra pa-1209'
	r'donated \w+',           # Donation information
	r'information received',  # Source information
	r'\w+ association',       # Organization suffixes without context
]

RELEVANT_ENTITY_TYPES = {
	'PERSON',
	'ORG',
	'GPE',
	'LOC',
	'NORP',
	'FAC',
	'EVENT',
	'WORK_OF_ART',
	'PRODUCT',
	'DATE',
	'TIME',
	'MISC',
}

def clean_text(text):
		"""Clean text by removing special characters and excess whitespace"""
		if not isinstance(text, str):
				return ""
		
		# Apply all metadata pattern removals
		for pattern in METADATA_PATTERNS:
				text = re.sub(pattern, '', text)

		# Replace specific patterns often found in metadata
		text = re.sub(r'\[\{.*?\}\]', '', text)  # Remove JSON-like structures
		text = re.sub(r'http\S+', '', text)      # Remove URLs
		text = re.sub(r'\d+\.\d+', '', text)     # Remove floating point numbers
		# Remove non-alphanumeric characters but keep spaces
		text = re.sub(r'[^\w\s]', ' ', text)
		# Replace multiple spaces with a single space
		text = re.sub(r'\s+', ' ', text)
		text = text.strip().lower()

		return text


def get_keywords(
		text: str, 
		sent_model: SentenceTransformer, 
		rake: Rake,
	):
	rake.extract_keywords_from_text(text)
	rake_phrases = [
		phrase 
		for phrase in rake.get_ranked_phrases() 
		if len(phrase.split()) <= 3 and phrase.lower() not in CUSTOM_STOPWORDS
	]

	kw_model = KeyBERT(model=sent_model)	
	keybert_keywords = kw_model.extract_keywords(
		text,
		keyphrase_ngram_range=(1, 3),
		stop_words=list(CUSTOM_STOPWORDS),
		top_n=20,
		use_mmr=True,  # Use Maximal Marginal Relevance for diversity
		diversity=0.7,
	)
	
	# Combine and rank by relevance to text
	all_candidates = list(set(rake_phrases[:15] + [kw[0] for kw in keybert_keywords]))
	if not all_candidates:
		print("No keywords extracted, returning empty list")
		return []

	# Filter and score candidates
	text_embedding = sent_model.encode(text, show_progress_bar=False)
	keyword_embeddings = sent_model.encode(all_candidates, show_progress_bar=False)
	if keyword_embeddings.size == 0 or text_embedding.size == 0:
		print("Empty keyword embeddings, returning empty list")
		return []
	similarities = np.dot(keyword_embeddings, text_embedding) / (np.linalg.norm(keyword_embeddings, axis=1) * np.linalg.norm(text_embedding) + 1e-8)
	
	# Dynamic threshold based on distribution
	threshold = np.percentile(similarities, 70)  # Keep top 30%
	filtered = [
		(cand, sim) 
		for cand, sim in zip(all_candidates, similarities) 
		if sim > threshold and is_likely_english_term(cand)
	]
	
	# Final selection with diversity
	selected = []
	used_words = set()
	for cand, sim in sorted(filtered, key=lambda x: x[1], reverse=True):
		words = set(cand.lower().split())
		overlap = len(words & used_words) / len(words)
		if overlap < 0.4:  # Allow some overlap but not too much
			selected.append(cand)
			used_words.update(words)
	return selected

def combine_and_clean_labels(
		ner_labels: List[str],
		keywords: List[str],
		topic_labels: List[str],
		user_query: Union[str, None],
		text: str,
		sent_model: SentenceTransformer,
		doc_year: Union[float, None] = None,
		relevance_threshold: float = 0.4,
		max_labels: int = 12,
		min_label_length: int = 4,
		semantic_coherence_threshold: float = 0.9
	) -> List[str]:

	def collect_weighted_labels(
			user_terms: List[str],
			ner_labels: List[str],
			keywords: List[str],
			topic_labels: List[str]
	) -> List[Tuple[str, float, str]]:
			weighted = []
			# User terms (high priority)
			for term in user_terms:
					weighted.append((term, 1.0, 'user'))
			# NER labels (favor proper nouns)
			for label in ner_labels:
					if len(label) >= min_label_length:
							weight = 0.9 if label[0].isupper() else 0.7
							weighted.append((label, weight, 'ner'))
			# Keywords (length-based weight)
			for label in keywords:
					if len(label) >= min_label_length:
							weight = 0.6 + 0.1 * len(label.split())
							weighted.append((label, weight, 'keyword'))
			# Topic labels (lowest priority)
			for label in topic_labels:
					if len(label) >= min_label_length:
							weighted.append((label, 0.4, 'topic'))
			return weighted
	def generate_embeddings(labels: List[Tuple[str, float, str]]) -> Tuple[List[str], np.ndarray]:
			label_texts = [lbl for lbl, _, _ in labels]
			if not label_texts:
					return [], np.array([])
			embeddings = sent_model.encode(
					label_texts,
					batch_size=64,
					show_progress_bar=False,
					convert_to_numpy=True
			)
			return label_texts, embeddings
	def perform_semantic_clustering(embeddings: np.ndarray) -> np.ndarray:
			if len(embeddings) <= 2:
					return np.zeros(len(embeddings), dtype=int)
			embeddings = normalize(embeddings, norm='l2', axis=1)
			min_cluster_size = max(2, len(embeddings) // 5)
			clusterer = hdbscan.HDBSCAN(
					min_cluster_size=min_cluster_size,
					min_samples=1,
					metric='euclidean',
					cluster_selection_method='eom'
			)
			return clusterer.fit_predict(embeddings)
	def process_clusters(
		clusters: np.ndarray,
		weighted_labels: List[Tuple[str, float, str]],
		label_to_emb: Dict[str, np.ndarray],
		text_emb: np.ndarray
		) -> List[str]:
		cluster_groups = defaultdict(list)
		for idx, (label, weight, source) in enumerate(weighted_labels):
			cluster_groups[clusters[idx]].append((label, weight, source))
		
		final_labels = []
		for cluster_id, members in cluster_groups.items():
			if cluster_id == -1:
				continue
			scored = []
			for label, weight, source in members:
					source_priority = {'user': 4, 'ner': 3, 'keyword': 2, 'topic': 1}[source]
					length_factor = 1.0 + 0.1 * len(label.split())
					sim = np.dot(label_to_emb[label], text_emb) / (
							np.linalg.norm(label_to_emb[label]) * np.linalg.norm(text_emb) + 1e-8
					)
					score = weight * length_factor * source_priority * (0.5 + 0.5 * sim)
					scored.append((score, label))
			if scored:
					top_score, top_label = max(scored)
					if top_score > relevance_threshold:
							final_labels.append(top_label)
		return final_labels
	def handle_noise_labels(
			clusters: np.ndarray,
			label_texts: List[str],
			embeddings: np.ndarray,
			text_emb: np.ndarray
	) -> List[str]:
			noise_mask = (clusters == -1)
			if not noise_mask.any():
					return []
			noise_labels = [label for label, mask in zip(label_texts, noise_mask) if mask]
			noise_embs = embeddings[noise_mask]
			sims = np.dot(noise_embs, text_emb) / (
					np.linalg.norm(noise_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
			)
			top_indices = np.argsort(sims)[-2:][::-1]
			return [noise_labels[i] for i in top_indices if sims[i] > relevance_threshold]
	def apply_final_filters(
			candidates: List[str],
			label_to_emb: Dict[str, np.ndarray],
			text_emb: np.ndarray
	) -> List[str]:
			if not candidates:
					return []
			final_embs = np.array([label_to_emb[label] for label in candidates])
			final_sims = np.dot(final_embs, text_emb) / (
					np.linalg.norm(final_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
			)
			filtered = []
			used_embeddings = []
			for label, sim in sorted(zip(candidates, final_sims), key=lambda x: -x[1]):
					if sim < relevance_threshold:
							continue
					label_emb = label_to_emb[label]
					if any(np.dot(label_emb, used_emb) / (
							np.linalg.norm(label_emb) * np.linalg.norm(used_emb) + 1e-8
					) > semantic_coherence_threshold for used_emb in used_embeddings):
							continue
					if any(fuzz.ratio(label.lower(), kept.lower()) > 90 for kept in filtered):
							continue
					filtered.append(label)
					used_embeddings.append(label_emb)
					if doc_year and not is_year_compatible(label, doc_year):
							filtered.remove(label)
					if len(filtered) >= max_labels:
							break
			return sorted(filtered)
	
	# 1. Parse and clean user query
	user_terms = parse_user_query(user_query)
	# 2. Collect weighted labels with source-based priority
	weighted_labels = collect_weighted_labels(user_terms, ner_labels, keywords, topic_labels)
	if not weighted_labels:
		return []
	# 3. Generate embeddings
	label_texts, embeddings = generate_embeddings(weighted_labels)
	if not embeddings.size:
		return []
	label_to_emb = {lbl: emb for lbl, emb in zip(label_texts, embeddings)}
	text_emb = sent_model.encode(text, show_progress_bar=False)
	# 4. Cluster embeddings
	clusters = perform_semantic_clustering(embeddings)
	# 5. Process clusters
	final_labels = process_clusters(clusters, weighted_labels, label_to_emb, text_emb)
	# 6. Handle noise labels
	final_labels += handle_noise_labels(clusters, label_texts, embeddings, text_emb)
	# 7. Final filtering and deduplication
	final_labels = apply_final_filters(final_labels, label_to_emb, text_emb)
	return final_labels


def balance_label_count(
		image_labels_list,
		text_descriptions,
		sent_model,
		min_labels=1,
		max_labels=12,
		similarity_threshold=0.5
	):
	balanced_labels = []
	
	# Encode text descriptions once
	text_embeds = sent_model.encode(text_descriptions, show_progress_bar=False, convert_to_numpy=True)
	
	for idx, labels in tqdm(enumerate(image_labels_list), total=len(image_labels_list), desc="Label Balancing"):
			text_emb = text_embeds[idx]
			
			# Case 1: Too few labels - generate coherent compounds
			if len(labels) < min_labels:
					compound_labels = []
					if len(labels) >= 2:
							# Encode existing labels
							label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
							for i in range(len(labels)):
									for j in range(i+1, len(labels)):
											compound = f"{labels[i]} {labels[j]}"
											if 2 <= len(compound.split()) <= 3:  # 2-3 word compounds
													# Compute similarity of compound to text
													compound_emb = sent_model.encode(compound, show_progress_bar=False)
													similarity = np.dot(compound_emb, text_emb) / (
															np.linalg.norm(compound_emb) * np.linalg.norm(text_emb) + 1e-8
													)
													if similarity > similarity_threshold:
															compound_labels.append(compound)
					
					# Sort compounds by similarity
					if compound_labels:
							compound_embs = sent_model.encode(compound_labels, show_progress_bar=False, convert_to_numpy=True)
							similarities = np.dot(compound_embs, text_emb) / (
									np.linalg.norm(compound_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
							)
							compound_labels = [compound_labels[i] for i in np.argsort(similarities)[::-1]]
					
					# Add compounds to reach min_labels
					expanded_labels = labels + compound_labels[:min_labels - len(labels)]
					balanced_labels.append(expanded_labels[:min_labels] if len(expanded_labels) >= min_labels else expanded_labels)
			
			# Case 2: Too many labels - prioritize multi-word by relevance
			elif len(labels) > max_labels:
					# Encode labels and compute similarities
					label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
					similarities = np.dot(label_embs, text_emb) / (
							np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
					)
					
					# Separate multi-word and single-word labels
					multi_word = [(label, sim) for label, sim in zip(labels, similarities) if ' ' in label]
					single_word = [(label, sim) for label, sim in zip(labels, similarities) if ' ' not in label]
					
					# Sort by similarity
					multi_word.sort(key=lambda x: x[1], reverse=True)
					single_word.sort(key=lambda x: x[1], reverse=True)
					
					# Prioritize multi-word, fill with single-word if needed
					selected_labels = [label for label, _ in multi_word[:max_labels]]
					if len(selected_labels) < max_labels:
							selected_labels.extend([label for label, _ in single_word[:max_labels - len(selected_labels)]])
					
					balanced_labels.append(selected_labels)
			
			# Case 3: Good label count - keep as is
			else:
					balanced_labels.append(labels)
	
	return balanced_labels

def get_all_unique_user_queries(df: pd.DataFrame) -> Set[str]:
		"""Collects all unique and cleaned user_query terms from the entire DataFrame."""
		unique_queries = set()
		if 'user_query' in df.columns:
				for query in df['user_query'].fillna('').astype(str):
						if query.strip():
								# Simple cleaning for user query before adding to global pool
								cleaned_query = re.sub(r'[^\w\s\-]', '', query.lower()).strip()
								if cleaned_query and cleaned_query not in CUSTOM_STOPWORDS and len(cleaned_query) > 2:
										unique_queries.add(cleaned_query)
		return unique_queries

def handle_multilingual_labels(labels):
	processed_labels = []
	for label in labels:
		# Skip non-ASCII labels completely
		if not all(ord(char) < 128 for char in label):
			continue
		words = label.split()
		# Single word label
		if len(words) == 1:
			if is_likely_english_term(label) or label[0].isupper():
				processed_labels.append(label)
		# Multi-word label
		else:
			# Keep if all words are likely English or proper nouns
			if all(is_likely_english_term(word) or word[0].isupper() for word in words):
				processed_labels.append(label)
	return processed_labels


def extract_named_entities(
	nlp: pipeline,
	text: str, 
	ft_model: fasttext.FastText._FastText,
	confidence_threshold: float = 0.7,
	) -> List[str]:
	if not text or not isinstance(text, str) or len(text) < 5 or not is_english(text, ft_model):
		return []
	print(text)
	try:
		ner_results = nlp(text)
		print(f"NER results ({len(ner_results)}): {ner_results}")
		entities = []
		seen_entities = set()
		for entity in ner_results:
			if entity.get("score", 0) < confidence_threshold or entity["entity_group"] not in RELEVANT_ENTITY_TYPES:
				continue
			full_entity = entity["word"].strip()
			full_entity = re.sub(r'^(The|A|An)\s+', '', full_entity, flags=re.I)
			full_entity = ' '.join(full_entity.split())
			if len(full_entity) < 3 or full_entity.lower() in CUSTOM_STOPWORDS:
				continue
			if not any(fuzz.ratio(full_entity.lower(), seen.lower()) > 95 for seen in seen_entities):
				normalized = full_entity if full_entity[0].isupper() else full_entity.lower()
				entities.append(normalized)
				seen_entities.add(full_entity.lower())
		try:
			tokens = nlp.tokenizer.tokenize(text)
			tokens = [token.lower() for token in tokens if token.lower() not in CUSTOM_STOPWORDS and len(token) > 2]
			meaningful_bigrams = [
				f"{tokens[i]} {tokens[i+1]}" 
				for i in range(len(tokens)-1) 
				if tokens[i] not in CUSTOM_STOPWORDS 
				and tokens[i+1] not in CUSTOM_STOPWORDS
				and len(tokens[i]) > 2 and len(tokens[i+1]) > 2
				and not any(f"{tokens[i]} {tokens[i+1]}".lower() in e.lower() for e in entities)
			]
			combined = list(set(entities + meaningful_bigrams))
			final_entities = []
			seen = set()
			for entity in sorted(combined, key=len, reverse=True):
				entity_lower = entity.lower()
				if not any(fuzz.ratio(entity_lower, s.lower()) > 95 for s in seen):
					final_entities.append(entity)
					seen.add(entity_lower)
			final_entities = sorted(final_entities)
			print(f"Final entities: {final_entities}")
			return final_entities
		except Exception as tokenize_error:
			print(f"Tokenization warning: {tokenize_error}")
			return sorted(list(set(entities)))
	except Exception as e:
		print(f"<!> NER error: {e} for text: {text}")
		return []


def extract_semantic_topics(
		sent_model: SentenceTransformer,
		ft_model: fasttext.FastText._FastText,
		texts: List[str],
		dataset_dir: str,
		num_workers: int,
		enable_visualizations: bool = False,
	) -> Tuple[List[List[str]], Set[str]]:
	
	vectorizer_model = CountVectorizer(
		ngram_range=(1, 3),
		stop_words=list(CUSTOM_STOPWORDS),
	)
	ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

	representation_model = {
		"KeyBERT": KeyBERTInspired(),
		"MMR": MaximalMarginalRelevance(diversity=0.5),
		# prefer nouns and adjectives
		"POS": PartOfSpeech("en_core_web_sm", pos_patterns=[[{"POS": "NOUN"}, {"POS": "ADJ"}]])
	}
	min_topic_size = max(10, min(20, len(texts)//500))
	print(f"Creating BERTopic model for {len(texts)} texts => min_topic_size: {min_topic_size}")
	topic_model = BERTopic(
		embedding_model=sent_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_model,
		min_topic_size=min_topic_size,
		calculate_probabilities=True,
		nr_topics="auto",
		verbose=True,
	)
	
	topics, probs = topic_model.fit_transform(texts)
	topic_info = topic_model.get_topic_info()
	print(f"Number of topics: {len(topic_info)}")
	print(topic_info)
	print(f"Unique Topic IDs: {topic_info['Topic'].unique()}")

	filtered_topics = []
	for topic_id in topic_info['Topic'].unique():
		if topic_id == -1:  # Skip outlier topic
			continue
		topic_words = [
			word 
			for word, score in topic_model.get_topic(topic_id) 
			if is_likely_english_term(word) and score > 0.05
		]
		if topic_words:
			filtered_topics.append(topic_words[:15])
	flat_topics = set(word for topic in filtered_topics for word in topic)
	return filtered_topics, flat_topics


def process_text_chunk(nlp, chunk):
	return [extract_named_entities(nlp=nlp, text=text) for text in chunk]

def parallel_relevance_filtering(texts, all_labels, n_processes=None):
	if n_processes is None:
		n_processes = max(1, multiprocessing.cpu_count() - 1)
	
	total_docs = len(texts)
	chunk_size = total_docs // n_processes + (1 if total_docs % n_processes else 0)
	chunks = []
	
	for i in range(0, total_docs, chunk_size):
		end_idx = min(i + chunk_size, total_docs)
		chunks.append((i, end_idx, texts[i:end_idx], all_labels[i:end_idx]))
	
	with multiprocessing.Pool(processes=n_processes) as pool:
		chunk_results = pool.map(process_document_chunk, chunks)
	
	all_results = []
	for chunk in chunk_results:
		all_results.extend(chunk)
	
	return all_results

def get_textual_based_annotation(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		relevance_threshold: float,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		ner_model_name: str,
		verbose: bool=True,
		use_parallel: bool=False,
	):
	if verbose:
		print(f"Automatic label extraction from text data".center(160, "-"))
		print(f"Loading metadata from {csv_file}...")
	text_based_annotation_start_time = time.time()
	dataset_dir = os.path.dirname(csv_file)
	
	if verbose:
		print(f"Loading sentence-transformer model: {st_model_name}...")

	sent_model = SentenceTransformer(model_name_or_path=st_model_name, device=device)
	ft_model = fasttext.load_model(FastText_Language_Identification)
	
	if verbose:
		print(f"Loading NER model: {ner_model_name}...")
	nlp = pipeline(
		task="ner", 
		model=AutoModelForTokenClassification.from_pretrained(ner_model_name),
		tokenizer=AutoTokenizer.from_pretrained(ner_model_name), 
		aggregation_strategy="simple",
		device=device,
		batch_size=batch_size,
	)

	dtypes = {
			'doc_id': str, 'id': str, 'label': str, 'title': str,
			'description': str, 'img_url': str, 'enriched_document_description': str,
			'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
			'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
			'user_query': str,
	}
	
	df = pd.read_csv(
			filepath_or_buffer=csv_file, 
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=False,
	)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
	
	# df['content'] = df['enriched_document_description'].fillna('').astype(str)
	df['content'] = df['enriched_document_description'].fillna('').astype(str)#.apply(preprocess_text)
	# Handle missing 'user_query' column
	if 'user_query' not in df.columns:
		if verbose:
			print("Warning: 'user_query' column missing in DataFrame. Using empty queries.")
		user_queries = [''] * len(df)
	else:
		user_queries = df['user_query'].fillna('').tolist()
	num_samples = df.shape[0]
	
	print(f"Filtering non-English entries for {num_samples} samples")
	t0 = time.time()
	english_mask = df['content'].apply(lambda x: is_english(text=x, ft_model=ft_model, verbose=False))
	english_indices = english_mask[english_mask].index.tolist()
	print(f"{sum(english_mask)} / {len(df)} texts are English [{sum(english_mask)/len(df)*100:.2f}%]")
	print(f"Elapsed_t: {time.time() - t0:.2f} sec")
	
	english_df = df[english_mask].reset_index(drop=True)
	english_texts = english_df['content'].tolist()
	english_queries = [user_queries[i] for i in english_indices]
	per_image_labels = [[] for _ in range(num_samples)]

	if len(english_texts) > 0:
		# Step 1: Topic Modeling
		print("Topic Modeling".center(160, "-"))
		t0 = time.time()
		topics, flat_topic_words = extract_semantic_topics(
			sent_model=sent_model,
			ft_model=ft_model,
			texts=english_texts,
			num_workers=num_workers,
			dataset_dir=dataset_dir,
		)
		print(f"{len(topics)} Topics (clusters) {type(topics)}:\n{[len(tp) for tp in topics]}")
		print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(160, "-"))
		
		# Step 2: Named Entity Recognition
		print("Extracting NER per sample...")
		t0 = time.time()
		if len(english_texts) > 1000 and use_parallel:
			chunk_size = len(english_texts) // num_workers + 1
			chunks = [(english_texts[i:i+chunk_size]) for i in range(0, len(english_texts), chunk_size)]
			print(f"Using {num_workers} processes for NER extraction...")
			with multiprocessing.Pool(processes=num_workers) as pool:
				ner_results = pool.map(process_text_chunk, [(nlp, chunk) for chunk in chunks])
			per_image_ner_labels = []
			for chunk_result in ner_results:
				per_image_ner_labels.extend(chunk_result)
		else:
			per_image_ner_labels = []
			for text in tqdm(english_texts, desc="NER Progress"):
				entities = extract_named_entities(nlp=nlp, text=text, ft_model=ft_model)
				per_image_ner_labels.append(entities)
		print(f"NER done in {time.time() - t0:.1f} sec")
		
		# Step 3: Extract keywords
		print("Extracting keywords per image using KeyBERT...")
		t0 = time.time()
		per_image_keywords = []
		rake = Rake(
			stopwords=list(CUSTOM_STOPWORDS),
			min_length=1,
			max_length=3,
			include_repeated_phrases=False
		)
		for text in tqdm(english_texts, desc="Keyword Extraction"):
			if not is_english(text, ft_model):
				per_image_keywords.append([])
				continue
			keywords = get_keywords(text, sent_model, rake)
			per_image_keywords.append(keywords)
		print(f"Keyword extraction done in {time.time() - t0:.1f} sec")

		# Step 4: Add topic labels
		print("Assigning topic labels per image...")
		t0 = time.time()
		per_image_topic_labels = []
		for text in tqdm(english_texts, desc="Topic Assignment"):
			matching_topics = [word for word in flat_topic_words if word in text.lower() and word not in CUSTOM_STOPWORDS]
			per_image_topic_labels.append(matching_topics)
		print(f"Topic assignment done in {time.time() - t0:.1f} sec")
		
		# Step 5: Combine and clean labels
		print("Combining and cleaning labels...")
		t0 = time.time()
		per_image_combined_labels = []
		for text, query, ner, keywords, topics in tqdm(zip(english_texts, english_queries, per_image_ner_labels, per_image_keywords, per_image_topic_labels), total=len(english_texts), desc="Label Combination"):
			cleaned_labels = combine_and_clean_labels(
				ner_labels=ner, 
				keywords=keywords, 
				topic_labels=topics, 
				user_query=query, 
				text=text, 
				sent_model=sent_model, 
				relevance_threshold=relevance_threshold,
			)
			per_image_combined_labels.append(cleaned_labels)
		print(f"Label combination and cleaning done in {time.time() - t0:.3f} sec")

		# Step 6: Filter by relevance
		print(f"Filtering labels by relevance (thresh: {relevance_threshold})...")
		t0 = time.time()
		if use_parallel:
			print("Using parallel processing for relevance filtering...")
			per_image_relevant_labels = parallel_relevance_filtering(
				texts=english_texts,
				all_labels=per_image_combined_labels,
				n_processes=num_workers,
			)
		else:
			print(f"Using batch processing for textual-based annotation ({batch_size} batches) for relevance filtering (thresh: {relevance_threshold})...")
			per_image_relevant_labels = batch_filter_by_relevance(
				sent_model=sent_model,
				texts=english_texts,
				all_labels_list=per_image_combined_labels,
				threshold=relevance_threshold,
				batch_size=batch_size,
			)
		print(f"Relevance filtering done in {time.time() - t0:.1f} sec")

		# Step 7: Post-process
		print(f"Post-processing of {len(per_image_relevant_labels)} textual labels, deduplication, and semantic categorization...")
		t0 = time.time()
		english_labels = []
		for i, relevant_labels in tqdm(enumerate(per_image_relevant_labels), total=len(per_image_relevant_labels), desc="Post-processing"):
			filtered_labels = handle_multilingual_labels(relevant_labels)
			filtered_labels = deduplicate_labels(filtered_labels)
			categorized = assign_semantic_categories(filtered_labels)
			final_labels = sorted(set(filtered_labels + categorized))
			english_labels.append(final_labels)
		print(f"Post-processing done in {time.time() - t0:.1f} sec")

		print("Balancing label counts...")
		t0 = time.time()
		english_labels = balance_label_count(
			image_labels_list=english_labels, 
			text_descriptions=english_texts, 
			sent_model=sent_model, 
			min_labels=1, 
			max_labels=12,
		)
		print(f"Label balancing done in {time.time() - t0:.3f} sec")
		
		# Transfer results
		for i, orig_idx in enumerate(english_indices):
			if i < len(english_labels):
				per_image_labels[orig_idx] = english_labels[i]
	else:
		print("No English texts found. Returning empty labels for all entries.")
	
	df['textual_based_labels'] = per_image_labels
	df.to_csv(metadata_fpth, index=False)
	
	print(f">> Generated text labels for {sum(1 for labels in per_image_labels if labels)} out of {num_samples} entries")
	print(f"Text-based annotation Elapsed time: {time.time() - text_based_annotation_start_time:.2f} sec".center(160, " "))
	
	return per_image_labels


def batch_filter_by_relevance_old(
		sent_model: SentenceTransformer,
		texts: List[str],
		all_labels_list: List[List[str]],
		threshold: float,
		batch_size: int,
	):
	results = []
	print("Pre-encoding texts embeddings...")
	try:
		text_embeddings = sent_model.encode(
			texts, 
			batch_size=batch_size,
			show_progress_bar=False,
			convert_to_numpy=True
		)
	except torch.cuda.OutOfMemoryError:
		print("CUDA Out of Memory. Fallback to smaller batches for text encoding...")
		text_embeddings = []
		for i in tqdm(range(0, len(texts), batch_size // 2), desc="Encoding texts (small batches)"):
			batch = texts[i:i + batch_size // 2]
			batch_emb = sent_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
			text_embeddings.extend(batch_emb)
		text_embeddings = np.array(text_embeddings)
	
	# Process labels efficiently
	for i, (text_emb, labels) in enumerate(tqdm(zip(text_embeddings, all_labels_list), total=len(texts), desc="Filtering labels")):
		if not labels:
			results.append([])
			continue
		
		# Dynamic batch size based on number of labels
		label_batch_size = min(len(labels), batch_size)
		if len(labels) > 100:  # Large label sets
			label_batch_size = batch_size // 4
		
		try:
			label_embeddings = sent_model.encode(
				labels, 
				batch_size=label_batch_size,
				show_progress_bar=False,
				convert_to_numpy=True
			)
			similarities = np.dot(label_embeddings, text_emb) / (np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8)			
			relevant_indices = np.where(similarities > threshold)[0]
			results.append([labels[idx] for idx in relevant_indices])
		except Exception as e:
			print(f"Error processing labels for text {i}: {e}")
			results.append([])
	return results

def deduplicate_labels_old(labels):
		"""
		Deduplicate labels based on semantic similarity and substring containment.
		
		Args:
				labels: List of labels
		
		Returns:
				List of deduplicated labels
		"""
		if not labels:
				return []
		
		# Substring and semantic deduplication
		deduplicated = []
		for label in sorted(labels, key=len, reverse=True):
				if not any(
						label in kept_label or kept_label in label or label.lower() == kept_label.lower()
						for kept_label in deduplicated
				):
						deduplicated.append(label)
		
		return sorted(deduplicated)


def clean_old(labels):
		cleaned = set()
		# Predefined list of valid historical non-ASCII terms
		VALID_HISTORICAL_TERMS = {
				"blitzkrieg", "kübelwagen", "wehrmacht", "panzer", "luftwaffe",
				"stuka", "t-34", "afrika korps"
		}
		for label in labels:
				# Normalize the label
				label = label.lower().strip()
				# Remove non-alphanumeric characters except spaces and hyphens
				label = re.sub(r"[^a-z0-9\s\-äöüßñéèêëìíîïçåæœø]", "", label)
				# Skip short labels, stopwords, or invalid labels
				if label in CUSTOM_STOPWORDS or len(label) < 3:
						continue
				# Skip labels that are just numbers
				if label.isdigit():
						continue
				# Skip labels that start with numbers unless they're years (4 digits)
				if label[0].isdigit() and not (len(label) == 4 and label.isdigit()):
						continue
				# Allow non-ASCII labels if proper noun or in valid historical terms
				if not all(ord(char) < 128 for char in label):
						if not (label[0].isupper() or label in VALID_HISTORICAL_TERMS):
								continue
				cleaned.add(label)
		return sorted(cleaned)

def combine_and_clean_labels(
		ner_labels: List[str],
		keywords: List[str],
		topic_labels: List[str],
		user_query: Union[str, None],
		text: str,
		sent_model: SentenceTransformer,
		relevance_threshold: float = 0.35,
		max_labels: int = 15,
		min_label_length: int = 3,
		semantic_coherence_threshold: float = 0.85
	) -> List[str]:
	# -- 1. Advanced User Query Parsing --
	user_terms = parse_user_query(user_query)
	
	# -- 2. Weighted Label Collection with Source Tracking --
	weighted_labels = collect_weighted_labels(
			user_terms,
			ner_labels,
			keywords,
			topic_labels,
			min_label_length
	)
	if not weighted_labels:
		return []

	# -- 3. Semantic Embedding and Clustering --
	label_texts, embeddings = generate_embeddings(weighted_labels, sent_model)
	clusters = perform_semantic_clustering(embeddings)
	
	# -- 4. Cluster Processing and Label Selection --
	label_to_emb = {label: emb for label, emb in zip(label_texts, embeddings)}
	text_emb = sent_model.encode(text, show_progress_bar=False)
	
	final_labels = process_clusters(
		clusters,
		weighted_labels,
		label_texts,
		label_to_emb,
		text_emb,
		relevance_threshold
	)
	
	# -- 5. Noise Processing --
	final_labels += handle_noise_labels(
		clusters,
		label_texts,
		embeddings,
		text_emb,
		relevance_threshold
	)
	
	# -- 6. Final Filtering and Deduplication --
	return apply_final_filters(
		final_labels,
		label_to_emb,
		text_emb,
		relevance_threshold,
		max_labels,
		semantic_coherence_threshold
	)

def parse_user_query(user_query: Union[str, None]) -> List[str]:
	if not user_query or not isinstance(user_query, str):
		return []
	
	try:
		# Try JSON parsing first
		if user_query.strip().startswith('['):
			parsed = json.loads(user_query.replace("'", '"'))
			return [str(t).strip() for t in parsed if str(t).strip()]
		
		# Try common delimiters
		for delim in [';', '|', ',', '\n']:
			if delim in user_query:
				return [t.strip() for t in user_query.split(delim) if t.strip()]
						
		# Fallback to single term
		return [user_query.strip()]
	except:
		return [user_query.strip()]

def collect_weighted_labels(
		user_terms: List[str],
		ner_labels: List[str],
		keywords: List[str],
		topic_labels: List[str],
		min_length: int
	) -> List[Tuple[str, float, str]]:
	"""Create weighted label collection with source-based weighting"""
	weighted = []
	
	# User terms get highest priority
	for term in user_terms:
		if len(term) >= min_length:
			weighted.append((term, 1.0, 'user'))
	
	# NER labels with case-sensitive weighting
	for label in ner_labels:
		if len(label) >= min_length:
			# Higher weight for proper nouns and longer phrases
			weight = 0.9 if label[0].isupper() else 0.7
			weight += 0.05 * min(3, len(label.split()))  # Bonus for multi-word
			weighted.append((label, weight, 'ner'))
	
	# Keywords with length-based weighting
	for label in keywords:
		if len(label) >= min_length:
			words = label.split()
			weight = 0.5 + (0.1 * len(words))  # 0.6 for 2 words, 0.7 for 3 words
			weighted.append((label, weight, 'keyword'))
	
	# Topic labels with base weight
	for label in topic_labels:
		if len(label) >= min_length:
			weighted.append((label, 0.4, 'topic'))

	return weighted

def generate_embeddings(
		weighted_labels: List[Tuple[str, float, str]],
		sent_model
	) -> Tuple[List[str], np.ndarray]:
	label_texts = [label for label, _, _ in weighted_labels]
	
	# Batch processing for efficiency
	batch_size = min(128, len(label_texts))
	embeddings = sent_model.encode(
		label_texts,
		batch_size=batch_size,
		show_progress_bar=False,
		convert_to_numpy=True
	)
	
	return label_texts, embeddings

def perform_semantic_clustering(
		embeddings: np.ndarray,
		min_cluster_size: int = 2,
		min_samples: int = 1
	) -> np.ndarray:
	# Normalize embeddings
	embeddings = normalize(embeddings, norm='l2', axis=1)
	
	# Adjust parameters based on input size
	n_samples = len(embeddings)
	min_cluster_size = min(min_cluster_size, max(2, n_samples // 5))
	min_samples = min(min_samples, max(1, n_samples // 10))
	
	# Handle edge cases
	if n_samples <= 2:
		return np.zeros(n_samples, dtype=int)  # All in one cluster
	
	try:
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			metric='euclidean',
			cluster_selection_method='eom',
			prediction_data=True,
			gen_min_span_tree=False  # Disable for small datasets
		)
		return clusterer.fit_predict(embeddings)
	except Exception as e:
		print(f"Clustering warning: {str(e)[:200]}")
		return np.zeros(n_samples, dtype=int)  # Fallback: all noise

def process_clusters(
		clusters: np.ndarray,
		weighted_labels: List[Tuple[str, float, str]],
		label_texts: List[str],
		label_to_emb: Dict[str, np.ndarray],
		text_emb: np.ndarray,
		min_similarity: float
) -> List[str]:
		"""Process clusters to select best representative labels"""
		cluster_groups = defaultdict(list)
		for idx, (label, weight, source) in enumerate(weighted_labels):
				cluster_id = clusters[idx]
				cluster_groups[cluster_id].append((label, weight, source))
		
		final_labels = []
		for cluster_id, members in cluster_groups.items():
				if cluster_id == -1:
						continue  # Noise handled separately
						
				# Score each candidate considering:
				# 1. Original weight
				# 2. Length (prefer multi-word)
				# 3. Source priority (user > ner > keyword > topic)
				# 4. Similarity to document text
				scored = []
				for label, weight, source in members:
						source_priority = {'user': 4, 'ner': 3, 'keyword': 2, 'topic': 1}[source]
						length_factor = 1.0 + 0.1 * len(label.split())
						
						# Calculate similarity to document
						sim = np.dot(label_to_emb[label], text_emb) / (
								np.linalg.norm(label_to_emb[label]) * np.linalg.norm(text_emb) + 1e-8
						)
						
						score = weight * length_factor * source_priority * (0.5 + 0.5 * sim)
						scored.append((score, label))
				
				# Take top scoring label that meets similarity threshold
				if scored:
						top_score, top_label = max(scored)
						if top_score > min_similarity:
								final_labels.append(top_label)
		
		return final_labels

def handle_noise_labels(
		clusters: np.ndarray,
		label_texts: List[str],
		embeddings: np.ndarray,
		text_emb: np.ndarray,
		min_similarity: float,
		max_noise_labels: int = 2
) -> List[str]:
		noise_mask = (clusters == -1)
		if not noise_mask.any():
				return []
		
		noise_labels = [label for label, mask in zip(label_texts, noise_mask) if mask]
		noise_embs = embeddings[noise_mask]
		
		# Normalized cosine similarity
		sims = np.dot(noise_embs, text_emb) / (
				np.linalg.norm(noise_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
		)
		
		# Get top labels meeting threshold
		top_indices = np.argsort(sims)[-max_noise_labels:][::-1]
		return [
				noise_labels[i] for i in top_indices
				if sims[i] > min_similarity
		]

def apply_final_filters(
		candidates: List[str],
		label_to_emb: Dict[str, np.ndarray],
		text_emb: np.ndarray,
		min_similarity: float,
		max_labels: int,
		semantic_threshold: float
) -> List[str]:
		"""Apply final semantic deduplication and filtering"""
		if not candidates:
				return []
		
		# Recalculate similarities for remaining candidates
		final_embs = np.array([label_to_emb[label] for label in candidates])
		final_sims = np.dot(final_embs, text_emb) / (
				np.linalg.norm(final_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
		)
		
		# Filter by similarity and remove duplicates
		filtered = []
		used_embeddings = []
		for label, sim in sorted(zip(candidates, final_sims), key=lambda x: -x[1]):
				if sim < min_similarity:
						continue
						
				# Semantic deduplication
				label_emb = label_to_emb[label]
				is_duplicate = False
				for used_emb in used_embeddings:
						similarity = np.dot(label_emb, used_emb) / (
								np.linalg.norm(label_emb) * np.linalg.norm(used_emb) + 1e-8
						)
						if similarity > semantic_threshold:
								is_duplicate = True
								break
								
				if not is_duplicate:
						filtered.append(label)
						used_embeddings.append(label_emb)
						if len(filtered) >= max_labels:
								break
								
		return filtered



def process_category_batch(
		batch_paths, batch_descriptions, batch_indices, df, categories, prompt_embeds,
		category_type, sent_model, processor, model, device, verbose, base_thresholds, sub_batch_size
):
		"""
		Process a batch of images for a specific category type with adaptive thresholding and sparse metadata handling.
		
		Args:
				batch_paths: List of image file paths
				batch_descriptions: List of enriched document descriptions
				batch_indices: List of global indices for the batch
				df: DataFrame with metadata
				categories: List of category strings
				prompt_embeds: Pre-computed prompt embeddings
				category_type: String ('object', 'scene', 'era', 'activity')
				sent_model: SentenceTransformer model
				processor: ALIGN processor
				model: ALIGN model
				device: Torch device
				verbose: Bool for logging
				base_thresholds: Dict of base thresholds per category type
				sub_batch_size: Size of sub-batches for GPU memory management
		
		Returns:
				batch_results: List of lists of selected categories
				batch_scores: List of lists of VLM scores
		"""
		valid_images = []
		valid_indices = []
		valid_descriptions = []
		failed_images = []

		# Handle sparse metadata with fallback to title
		user_queries = df['user_query'].fillna('').iloc[batch_indices].tolist() if 'user_query' in df.columns else [''] * len(batch_indices)
		titles = df['title'].fillna('').iloc[batch_indices].tolist() if 'title' in df.columns else [''] * len(batch_indices)

		# Validate images and descriptions
		for i, path in enumerate(batch_paths):
			try:
				if os.path.exists(path):
					img = Image.open(path).convert('RGB')
					desc = batch_descriptions[i].strip()
					if not desc and user_queries[i].strip():
						desc = user_queries[i]
					if not desc and titles[i].strip():
						desc = titles[i]
					if len(desc.strip()) < 10:
						desc = ""  # Treat as sparse
					valid_images.append(img)
					valid_indices.append(i)
					valid_descriptions.append(desc)
				else:
					failed_images.append(path)
			except Exception as e:
				failed_images.append(path)
				if verbose:
					print(f"Error loading image {path}: {e}")

		if not valid_images:
			if verbose and failed_images:
				print(f"Failed to load {len(failed_images)} images in batch")
			return [[] for _ in range(len(batch_paths))], [[] for _ in range(len(batch_paths))]

		# Add dynamic categories from user_query
		extended_categories = categories.copy()
		new_queries = []
		for query in user_queries:
			if isinstance(query, str) and query.strip() and query not in extended_categories:
				# extended_categories.append(query)
				# new_queries.append(query)
				try:
					# Try to parse as a list
					parsed_query = ast.literal_eval(query) if query.startswith('[') and query.endswith(']') else [query]
					for q in parsed_query:
						if isinstance(q, str) and q.strip() and q not in extended_categories:
							extended_categories.append(q)
							new_queries.append(q)
				except (ValueError, SyntaxError):
					# Treat as single string if parsing fails
					if query not in extended_categories:
						extended_categories.append(query)
						new_queries.append(query)

		# Update prompt embeddings
		extended_prompt_embeds = prompt_embeds
		if new_queries:
			new_prompts = [f"a photo of {q}" for q in new_queries]
			new_embeds = sent_model.encode(new_prompts, device=device, convert_to_tensor=True, show_progress_bar=False)
			extended_prompt_embeds = torch.cat([prompt_embeds, new_embeds], dim=0)

		# Compute text similarities
		text_prompts = [f"a photo of {cat}" for cat in extended_categories]
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			text_embeds = sent_model.encode(valid_descriptions, device=device, convert_to_tensor=True, show_progress_bar=False)
			text_similarities = torch.mm(extended_prompt_embeds, text_embeds.T).cpu().numpy()
			prompt_norms = torch.norm(extended_prompt_embeds, dim=1, keepdim=True)
			text_norms = torch.norm(text_embeds, dim=1, keepdim=True)
			text_similarities = text_similarities / (torch.mm(prompt_norms, text_norms.T).cpu().numpy() + 1e-8)
			model.eval()
			batch_results = [[] for _ in range(len(batch_paths))]
			batch_scores = [[] for _ in range(len(batch_paths))]
			# Process sub-batches
			for sub_idx in range(0, len(valid_images), sub_batch_size):
				sub_end = min(sub_idx + sub_batch_size, len(valid_images))
				sub_images = valid_images[sub_idx:sub_end]
				sub_valid_indices = valid_indices[sub_idx:sub_end]
				inputs = processor(
					text=text_prompts,
					images=sub_images,
					padding="max_length",
					return_tensors="pt",
				).to(device)
				outputs = model(**inputs)
				image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
				vlm_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
				vlm_similarity = (100.0 * image_embeds @ vlm_embeds.T).softmax(dim=-1).cpu().numpy()
				threshold = base_thresholds[category_type]
				for i, img_idx in enumerate(range(sub_idx, sub_end)):
					batch_idx = sub_valid_indices[i]
					local_img_idx = img_idx - sub_idx
					# Entropy-based threshold adjustment
					img = sub_images[i]
					img_array = np.array(img.convert('L'))
					img_array = resize(img_array, (128, 128), anti_aliasing=True, preserve_range=True).astype(np.uint8)
					entropy = shannon_entropy(img_array)
					complexity_factor = max(0.7, 1.0 - 0.15 * (entropy / 8.0))
					adaptive_threshold = max(0.15, threshold * complexity_factor)
					is_sparse = not valid_descriptions[img_idx].strip()
					for cat_idx, cat in enumerate(extended_categories):
						# Moved text_score computation inside the loop
						text_score = text_similarities[cat_idx, img_idx]
						adjusted_threshold = adaptive_threshold
						if text_score > 0.5:
							adjusted_threshold *= 0.8
						elif text_score < 0.1:
							adjusted_threshold *= 1.2
						vlm_score = vlm_similarity[local_img_idx, cat_idx]
						combined_score = (
							0.5 * vlm_score + 0.5 * text_score if is_sparse
							else vlm_score * (0.8 + 0.2 * text_score)
						)
						if combined_score > adjusted_threshold:
							batch_results[batch_idx].append(cat)
							batch_scores[batch_idx].append(vlm_score)

		return batch_results, batch_scores


def post_process_labels(labels, text_description, sent_model, doc_year, vlm_scores, max_labels=10, similarity_threshold=0.8):
		"""
		Post-process visual labels to remove duplicates, validate temporally, and rank by relevance.
		
		Args:
				labels: List of labels
				text_description: String enriched document description
				sent_model: SentenceTransformer model
				doc_year: Float or int document year
				vlm_scores: List of VLM similarity scores
				max_labels: Maximum number of labels to retain
				similarity_threshold: Cosine similarity threshold for deduplication
		
		Returns:
				List of processed labels
		"""
		if not labels:
				return []
		
		# Validate vlm_scores length
		if len(vlm_scores) != len(labels):
				vlm_scores = vlm_scores + [0.0] * (len(labels) - len(vlm_scores)) if len(vlm_scores) < len(labels) else vlm_scores[:len(labels)]
		
		# Encode labels and text
		label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
		text_emb = sent_model.encode(text_description, show_progress_bar=False, convert_to_numpy=True)
		
		# Deduplicate based on semantic similarity
		deduplicated = []
		for i, (label, score) in enumerate(zip(labels, vlm_scores)):
				is_redundant = False
				for kept_label, kept_emb, _ in deduplicated:
						sim = np.dot(label_embs[i], kept_emb) / (np.linalg.norm(label_embs[i]) * np.linalg.norm(kept_emb) + 1e-8)
						if sim > similarity_threshold:
								is_redundant = True
								break
				if not is_redundant:
						deduplicated.append((label, label_embs[i], score))
		
		# Temporal validation
		validated = [
				(label, emb, score)
				for label, emb, score in deduplicated
				if is_year_compatible(label, doc_year)
		]
		
		# Rank by combined VLM and text similarity
		if validated:
				text_sims = np.dot(np.array([emb for _, emb, _ in validated]), text_emb) / (
						np.linalg.norm([emb for _, emb, _ in validated], axis=1) * np.linalg.norm(text_emb) + 1e-8
				)
				combined_scores = [0.6 * vlm_score + 0.4 * text_sim for vlm_score, text_sim in zip(
						[score for _, _, score in validated], text_sims
				)]
				ranked = [label for _, label in sorted(zip(combined_scores, [label for label, _, _ in validated]), reverse=True)]
				return ranked[:max_labels]
		
		return []

def combine_and_clean_labels(
		ner_labels: List[str], 
		keywords: List[str], 
		topic_labels: List[str], 
		user_query: str, 
		text: str, 
		sent_model: SentenceTransformer,
		relevance_threshold: float = 0.35
	):
	
	# Parse user query more robustly
	user_terms = []
	if user_query and isinstance(user_query, str):
		try:
			if user_query.startswith('[') and user_query.endswith(']'):
				parsed = ast.literal_eval(user_query)
				user_terms = [str(term).strip().lower() for term in parsed if str(term).strip()]
			else:
				# Split on common delimiters
				user_terms = [term.strip().lower() for term in re.split(r'[,;|]', user_query) if term.strip()]
		except:
			user_terms = [user_query.strip().lower()]
	
	# Combine all label sources with weights
	weighted_labels = []
	
	# Higher weight for user queries (most reliable)
	for term in user_terms:
		if term and len(term) > 2 and term not in CUSTOM_STOPWORDS:
			weighted_labels.append((term, 1.0, 'user'))
	
	# Medium weight for NER (reliable entities)
	for label in ner_labels:
		if label and len(label) > 2:
			weighted_labels.append((label, 0.8, 'ner'))
	
	# Lower weight for keywords and topics
	for label in keywords:
		if label and len(label) > 2:
			weighted_labels.append((label, 0.6, 'keyword'))
	
	for label in topic_labels:
		if label and len(label) > 2:
			weighted_labels.append((label, 0.4, 'topic'))
	
	if not weighted_labels:
		return []
	
	# Semantic clustering to group similar labels
	labels_only = [label for label, _, _ in weighted_labels]
	label_embeddings = sent_model.encode(labels_only, show_progress_bar=False)
	text_embedding = sent_model.encode(text, show_progress_bar=False)
	
	# Calculate relevance to original text
	relevance_scores = np.dot(label_embeddings, text_embedding) / (np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_embedding) + 1e-8)
	
	# Filter by relevance and deduplicate semantically
	final_labels = []
	used_embeddings = []
	
	# Sort by combined score (weight * relevance)
	combined_scores = [
		(label, weight * relevance_scores[i], source) 
		for i, (label, weight, source) in enumerate(weighted_labels)
	]
	combined_scores.sort(key=lambda x: x[1], reverse=True)
	
	for label, score, source in combined_scores:
		if score < relevance_threshold:
			continue

		# Check semantic similarity with already selected labels
		label_emb = sent_model.encode(label, show_progress_bar=False)
		is_redundant = False
		
		for used_emb in used_embeddings:
			similarity = np.dot(label_emb, used_emb) / (np.linalg.norm(label_emb) * np.linalg.norm(used_emb) + 1e-8)
			if similarity > 0.85:  # High similarity threshold
				is_redundant = True
				break
		
		if not is_redundant:
			final_labels.append(label)
			used_embeddings.append(label_emb)
	
	return final_labels



def density_based_parameters(embeddings, n_neighbors=15):
		"""Determine parameters based on local density"""
		# Compute nearest neighbors distances
		nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
		distances, _ = nbrs.kneighbors(embeddings)
		
		# Estimate density
		kth_distances = distances[:, -1]
		density = 1.0 / (kth_distances + 1e-8)
		
		# Use percentiles of density to determine parameters
		min_cluster_size = int(np.percentile(density, 75) * len(embeddings) / 100)
		min_samples = int(np.percentile(density, 50) * len(embeddings) / 100)
		
		# Ensure reasonable bounds
		min_cluster_size = max(10, min(min_cluster_size, len(embeddings)//10))
		min_samples = max(5, min(min_samples, min_cluster_size//2))
		
		return min_cluster_size, min_samples

def find_knee_point(embeddings, max_size=300):
		"""Find knee point in k-distance graph"""
		# Sample a subset for efficiency
		sample_size = min(5000, len(embeddings))
		sample = embeddings[np.random.choice(len(embeddings), sample_size, replace=False)]
		
		# Compute k-nearest neighbors distances
		n_neighbors = min(15, sample_size-1)
		knn = NearestNeighbors(n_neighbors=n_neighbors)
		knn.fit(sample)
		distances, _ = knn.kneighbors(sample)
		
		# Sort distances
		k_distances = np.sort(distances[:, -1])[::-1]
		
		# Find knee point
		kneedle = KneeLocator(
				range(len(k_distances)),
				k_distances,
				curve='convex',
				direction='decreasing'
		)
		
		return max(2, int(kneedle.knee * len(embeddings) / sample_size))

def find_optimal_min_cluster_size(embeddings, dataset_size, max_clusters=50):
		"""Find optimal min_cluster_size using silhouette analysis"""
		candidate_sizes = [
				int(np.sqrt(dataset_size)),  # Current approach
				int(np.log(dataset_size)**2),  # Log-based
				100, 150, 200, 250,  # Common values
				int(dataset_size**0.4),  # Alternative power law
		]
		
		best_score = -1
		best_size = candidate_sizes[0]
		
		for size in sorted(set(candidate_sizes)):
				if size < 2:
						continue
						
				# Cap at reasonable max clusters
				if dataset_size / size > max_clusters:
						continue
						
				clusterer = hdbscan.HDBSCAN(
						min_cluster_size=size,
						min_samples=max(1, int(size/2)),
						cluster_selection_method='leaf',
						metric='euclidean'
				)
				labels = clusterer.fit_predict(embeddings)
				
				# Only calculate silhouette score if we have multiple clusters
				n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
				if n_clusters > 1:
						sample_size = min(10000, len(embeddings))
						sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
						score = silhouette_score(embeddings[sample_indices], labels[sample_indices])
						
						if score > best_score:
								best_score = score
								best_size = size
		
		return best_size

def get_hdbscan_parameters(
		embeddings, 
		use_static=False, 
		minimum_cap=10,
		percentage=None,
	):
	print(f"get hdbscan parameters for embeddings {embeddings.shape}...".center(160, "-"))
	if use_static:
			return 100, 10
	num_samples, num_embeddings = embeddings.shape
	# Method 1: Silhouette-based
	silhouette_size = find_optimal_min_cluster_size(embeddings, num_samples)
	
	# Method 2: Knee point detection
	knee_size = find_knee_point(embeddings)
	
	# Method 3: Density-based
	density_size, density_samples = density_based_parameters(embeddings)
	
	# Dynamically set percentage based on dataset size
	if num_samples < 5000:
			percentage = 0.005  # 0.5% for small datasets
	elif num_samples < 50000:
			percentage = 0.0007  # 0.07% for medium datasets (increased from 0.0005)
	else:
			percentage = 0.0003  # 0.03% for large datasets
	
	percentage_size = max(minimum_cap, int(num_samples * percentage))
	# Combine results (for logging)
	suggestions = [
			silhouette_size,
			knee_size,
			density_size,
			percentage_size,
			int(np.sqrt(num_samples) // 2),
			int(np.sqrt(num_samples) / 2),
			int(np.log(num_samples)**2),
			int(np.log(num_samples)**3),
			50, 70, 80, 100, 120,
	]
	print(f"Suggestions: {suggestions}")
	
	# Remove outliers (for logging)
	q25, q75 = np.percentile(suggestions, [25, 75])
	iqr = q75 - q25
	print(f"q25: {q25}, q75: {q75}, iqr: {iqr}")
	filtered = [x for x in suggestions if (x >= q25 - 2*iqr) and (x <= q75 + 2*iqr)]
	print(f"Filtered: {filtered} median: {np.median(filtered)} min: {min(filtered)} max: {max(filtered)} mean: {np.mean(filtered)}")
	
	# Use percentage-based size directly
	min_cluster_size = percentage_size
	
	# Adjust min_samples to scale with min_cluster_size
	min_samples = max(3, int(min_cluster_size * 0.1))  # 10% of min_cluster_size, min 3
	
	# Log expected number of clusters
	expected_clusters = num_samples / min_cluster_size
	print(f"Expected number of clusters: {expected_clusters:.0f}")
	
	return min_cluster_size, min_samples

def extract_semantic_topics_old(
		sent_model: SentenceTransformer,
		ft_model: fasttext.FastText._FastText,
		texts: List[str],
		dataset_dir: str,
		num_workers: int,
		enable_visualizations: bool = True,
	) -> Tuple[List[List[str]], Set[str]]:

	# Generate embeddings
	kw_model = KeyBERT(model=sent_model)
	dataset_size = len(texts)
	sentence_transformer_name = sent_model._first_module().auto_model.config._name_or_path.replace(r"/", "_") # 'sentence-transformers_all-mpnet-base-v2'
	emb_fpth = os.path.join(dataset_dir, f'{sentence_transformer_name}_embeddings_{dataset_size}_samples.gz')
	t0 = time.time()
	try:
		embeddings = load_pickle(fpath=emb_fpth)
	except Exception as e:
		print(e)
		print(f"Generating Text embeddings for {len(texts)} texts [might take a while]...")
		embeddings = sent_model.encode(texts, show_progress_bar=True)
		save_pickle(pkl=embeddings, fname=emb_fpth)

	print(f"Raw Embeddings: {embeddings.shape} generated in {time.time() - t0:.2f} sec")
	small_dataset_sample_size = min(5000, dataset_size)
	t0 = time.time()
	if dataset_size < small_dataset_sample_size:
		print("Dataset is small, using KMeans for clustering...")
		kmeans = KMeans(n_clusters=min(10, max(2, int(np.sqrt(dataset_size)))), random_state=42)
		labels = kmeans.fit_predict(embeddings)
	else:
		print(f"Dataset is large: {dataset_size} samples => HDBSCAN clustering...")
		min_cluster_size, min_samples = get_hdbscan_parameters(
			embeddings=embeddings,
			use_static=False,
		)
		# Check for high noise in a sample of the data
		print(f"Checking for high noise in a sample of the data...")
		sample_size = small_dataset_sample_size
		sample_indices = np.random.choice(dataset_size, sample_size, replace=False)
		sample_embeddings = embeddings[sample_indices]
		sample_clusterer = hdbscan.HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			cluster_selection_method='eom',
			metric='euclidean',
		)
		sample_labels = sample_clusterer.fit_predict(sample_embeddings)
		noise_ratio = np.sum(sample_labels == -1) / sample_size
		print(f"Initial noise ratio ({sample_size} samples): {noise_ratio:.2%}")
		if noise_ratio > 0.3:  # Threshold for applying UMAP
			print(f"High noise detected ({noise_ratio:.2%}), applying UMAP preprocessing...")
			# UMAP embedding:
			print(f"Reducing embeddings: {embeddings.shape} to 50D for clustering using UMAP...")
			umap_reducer = umap.UMAP(
				n_neighbors=5,
				min_dist=0.1,
				densmap=True,
				spread=1.0,
				n_components=50, 
				random_state=42, 
				metric='cosine',
				n_jobs=num_workers,
			)
			embeddings = umap_reducer.fit_transform(embeddings)
			print(f"UMAP embedding {embeddings.shape} generated in {time.time() - t0:.2f} sec")
		print(f"Clustering embeddings {embeddings.shape} into topics with HDBSCAN...")
		cluster_selection_method = 'eom' if dataset_size < 50000 else 'leaf'
		print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, cluster_selection_method(dataset_size: {dataset_size}): {cluster_selection_method}")
		clusterer = hdbscan.HDBSCAN(
			alpha=1.0,
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			algorithm='best',
			metric='euclidean',
			cluster_selection_method=cluster_selection_method,
			core_dist_n_jobs=num_workers,
		)
		labels = clusterer.fit_predict(embeddings)

	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points (-1)
	print(f">>>> Found {n_clusters} clusters (excluding noise points) in {time.time() - t0:.2f} sec")
	print(f"Cluster Noise (-1) contains: {np.sum(labels == -1)} samples, [{np.sum(labels == -1) / len(labels) * 100:.2f}%]")

	# Visualization 1: Cluster Distribution Bar Plot (Before Merging)
	if enable_visualizations:
		topic_counts = Counter(labels)
		plt.figure(figsize=(16, 7))
		bars = plt.bar(range(len(topic_counts)), [topic_counts[i] for i in sorted(topic_counts.keys())], color='#3785e6', label='Before Merging')
		plt.title('Sample Distribution Across Clusters (Before Merging) [Noise included]')
		plt.xlabel('Cluster ID')
		plt.ylabel('Number of Documents')
		plt.xticks(range(len(topic_counts)), labels=sorted(topic_counts.keys()), fontsize=8, va='center', ha='center', rotation=90)
		for bar in bars:
			yval = bar.get_height()
			plt.text(bar.get_x() + bar.get_width()/2, yval + 0.7, int(yval), ha='center', va='bottom', fontsize=7, color='#16171a')
		plt.legend()
		plt.savefig(os.path.join(dataset_dir, f'topic_distribution_before_merging_{n_clusters}_clusters.png'), bbox_inches='tight')
		plt.close()

	# # Visualization 2: Interactive UMAP Scatter Plot with Plotly
	# if enable_visualizations:
	# 	print(f"UMAP reducing embeddings: {embeddings.shape}")
	# 	umap_reducer = umap.UMAP(
	# 		n_neighbors=15,
	# 		min_dist=0.1,
	# 		densmap=True,
	# 		spread=1.0,
	# 		n_components=2, 
	# 		random_state=42, 
	# 		metric='cosine',
	# 	)
	# 	emb_umap = umap_reducer.fit_transform(embeddings)

	# 	centroids = np.zeros((n_clusters, 2))
	# 	for i in range(n_clusters):
	# 		cluster_points = emb_umap[labels == i]
	# 		if len(cluster_points) > 0:
	# 			centroids[i] = np.mean(cluster_points, axis=0)
	# 	distances = np.array([np.linalg.norm(emb_umap[i] - centroids[labels[i]]) if labels[i] != -1 else 0 for i in range(len(texts))])
	# 	outliers = distances > (np.mean(distances[distances > 0]) + 2 * np.std(distances[distances > 0])) if distances[distances > 0].size > 0 else np.zeros(len(texts), dtype=bool)
	# 	df_plot = pd.DataFrame({
	# 		'UMAP1': emb_umap[:, 0],
	# 		'UMAP2': emb_umap[:, 1],
	# 		'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels],
	# 		'Text': [text[:100] + '...' if len(text) > 100 else text for text in texts],
	# 		'Distance_to_Centroid': distances,
	# 		'Outlier': ['Yes' if o else 'No' for o in outliers]
	# 	})
	# 	fig = px.scatter(
	# 		df_plot,
	# 		x='UMAP1',
	# 		y='UMAP2',
	# 		color='Cluster',
	# 		symbol='Outlier',
	# 		hover_data=['Text', 'Distance_to_Centroid'],
	# 		title=f'Interactive UMAP Visualization of Text Embeddings for {dataset_size} Texts into {n_clusters} Cluster'
	# 	)
	# 	fig.add_trace(go.Scatter(
	# 		x=centroids[:, 0],
	# 		y=centroids[:, 1],
	# 		mode='markers+text',
	# 		marker=dict(size=15, symbol='x', color='#000000'),
	# 		text=[f'Centroid {i}' for i in range(n_clusters)],
	# 		textposition='top center',
	# 		name='Centroids'
	# 	))
	# 	fig.write_html(os.path.join(dataset_dir, 'umap_cluster_visualization_interactive.html'))

	# Collect phrases for each cluster
	print("Extracting keywords for each cluster using KeyBERT...")
	
	t0 = time.time()
	cluster_phrases = defaultdict(Counter)
	cluster_text_counts = defaultdict(int)
	phrase_filter_log = {'total_phrases': 0, 'stopword_filtered': 0, 'length_filtered': 0}
	for i, (text, label) in tqdm(enumerate(zip(texts, labels)), total=len(texts), desc="Keywords Extraction Progress"):
		if label == -1:  # Skip noise points
			continue
		phrases = kw_model.extract_keywords(
			text,
			keyphrase_ngram_range=(1, 3),
			stop_words="english",
			top_n=15 if len(text.split()) > 100 else 5,
			diversity=0.7,
		)
		# print(phrases)
		phrase_filter_log['total_phrases'] += len(phrases)
		# Filter phrases
		valid_phrases = []
		for phrase, _ in phrases:  # Ignore KeyBERT scores for now
			words = phrase.split()
			stopword_count = sum(1 for word in words if word in CUSTOM_STOPWORDS)
			# Relaxed stopword filter: allow <=70% stopwords
			if stopword_count / len(words) > 0.7:
				phrase_filter_log['stopword_filtered'] += 1
				continue
			valid_phrases.append(phrase)
		
		# Normalize phrases to reduce repetition
		normalized_phrases = []
		seen_phrases = set()
		for phrase in valid_phrases:
			# Remove consecutive duplicate words
			words = phrase.split()
			normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
			# Ensure normalized phrase is unique and meets length requirement
			if len(normalized.split()) >= 1 and normalized not in seen_phrases:
				normalized_phrases.append(normalized)
				seen_phrases.add(normalized)
			else:
				phrase_filter_log['length_filtered'] += 1
		
		if not normalized_phrases:
			print(f"Text {i} has no valid phrases: {text}")
		else:
			cluster_phrases[label].update(normalized_phrases)
		cluster_text_counts[label] += 1
	print(f"Phrase collection done in {time.time() - t0:.2f} sec")

	# Visualization 3: Phrase Retention Histogram
	if enable_visualizations:
		print("Phrase Retention Histogram")
		plt.figure(figsize=(12, 6))
		plt.bar(
			['Total Phrases', 'Stopword Filtered', 'Length Filtered'], 
			[phrase_filter_log['total_phrases'], 
			phrase_filter_log['stopword_filtered'], 
			phrase_filter_log['length_filtered']],
			color=['#005dcf', '#df7272', '#8dfc8d']
		)
		for i, val in enumerate([phrase_filter_log['total_phrases'], phrase_filter_log['stopword_filtered'], phrase_filter_log['length_filtered']]):
			plt.text(i, val + 0.5, str(val), ha='center', va='bottom')
		plt.title('Phrase Retention After Filtering')
		plt.ylabel('Number of Phrases')
		plt.savefig(os.path.join(dataset_dir, 'phrase_retention_histogram.png'), bbox_inches='tight')
		plt.close()
	
	# Extract initial topics with diversity scoring
	topics_before_merging = []
	term_counts_per_cluster = []
	for label, counter in cluster_phrases.items():
		if not counter:
			print(f"Warning: Topic {label} has no phrases.")
		# Score phrases with diversity bonus
		phrase_scores = []
		top_k_words = max(10, len(counter) // 15)
		# print(f"Topic {label}: {len(counter)} phrases, Selecting Top-{top_k_words}")
		seen_words = set()
		for phrase, count in counter.items():
			words = set(phrase.split())
			diversity_bonus = sum(1 for word in words if word not in seen_words)
			score = count * (1 + 0.1 * len(words) + 0.7 * diversity_bonus)
			phrase_scores.append((phrase, count, score))
			seen_words.update(words)
		phrase_scores.sort(key=lambda x: x[2], reverse=True)
		selected_phrases = []
		seen_words = set()
		for phrase, count, score in phrase_scores[:top_k_words * 2]:
			words = set(phrase.split())
			if any(words.issubset(set(p.split())) and counter[p] > count * 2 for p in selected_phrases):
				continue
			selected_phrases.append(phrase)
			seen_words.update(words)
		topics_before_merging.append(selected_phrases[:top_k_words])
		term_counts_per_cluster.append(len(counter))
	if not any(topics_before_merging):
		print("Error: No valid phrases found in any topics.")
		return [], set()
	
	# Calculate topic similarities
	print("Calculating topic similarities for merging [cosine similarity]...")
	similarity_matrix = np.zeros((len(topics_before_merging), len(topics_before_merging)))
	word_to_embedding = {}
	all_words = list(set(word for topic in topics_before_merging for word in topic if word))
	if all_words:
		word_embeddings = sent_model.encode(all_words, show_progress_bar=True)
		print(f"Word embeddings shape: {word_embeddings.shape}")
		for i, word in enumerate(all_words):
			word_to_embedding[word] = word_embeddings[i]
	else:
		print("Warning: No words available for topic embeddings.")

	topic_embeddings = []
	for topic in topics_before_merging:
		topic_embs = [word_to_embedding[word] for word in topic if word in word_to_embedding]
		topic_emb = np.mean(topic_embs, axis=0) if topic_embs else np.zeros(word_embeddings.shape[1])
		topic_embeddings.append(topic_emb)
	for i in range(len(topics_before_merging)):
		for j in range(i + 1, len(topics_before_merging)):
			sim = util.cos_sim([topic_embeddings[i]], [topic_embeddings[j]])[0][0].item()
			similarity_matrix[i, j] = sim
			similarity_matrix[j, i] = sim

	# Visualization 4: Phrase Co-Occurrence Network for Each Topic [before merging]
	if enable_visualizations:
		print(f"Generating co-occurrence networks for {len(topics_before_merging)} topics [before merging]...")
		for label, topic_phrases in enumerate(topics_before_merging):
			if not topic_phrases:
				# print(f"Skipping Topic {label}: No phrases available.")
				continue
			# Select top 10 phrases by frequency to reduce clutter
			counter = cluster_phrases[label]
			phrase_freq = {phrase: counter[phrase] for phrase in topic_phrases if phrase in counter}
			top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
			top_phrases = [phrase for phrase, _ in top_phrases]
			if not top_phrases:
				# print(f"Skipping Topic {label}: No phrases after filtering.")
				continue
			cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label and is_english(texts[i], ft_model)]
			if not cluster_texts:
				# print(f"Skipping Topic {label}: No valid texts for co-occurrence.")
				continue
			phrase_set = set(top_phrases)
			cooc_matrix = defaultdict(int)
			for text in cluster_texts:
				phrases = kw_model.extract_keywords(
					text,
					keyphrase_ngram_range=(1, 3),
					stop_words="english",
					top_n=15 if len(text.split()) > 100 else 5,
					diversity=0.7,
				)
				valid_phrases = []
				seen_phrases = set()
				for phrase, _ in phrases:
					words = phrase.split()
					stopword_count = sum(1 for w in words if w in CUSTOM_STOPWORDS)
					if stopword_count / len(words) > 0.6 or len(words) < 2:
						continue
					normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
					if len(normalized.split()) >= 2 and normalized not in seen_phrases:
						valid_phrases.append(normalized)
						seen_phrases.add(normalized)
				text_phrases = set(valid_phrases).intersection(phrase_set)
				for p1 in text_phrases:
					for p2 in text_phrases:
						if p1 < p2:
							cooc_matrix[(p1, p2)] += 1
			G = nx.Graph()
			for (p1, p2), count in cooc_matrix.items():
				if count >= 2: # An edge exists between two phrases if they appear together in the same text at least twice
					G.add_edge(p1, p2, weight=count)
			for phrase in top_phrases:
				if phrase not in G:
					G.add_node(phrase)
			plt.figure(figsize=(12, 8))
			pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjusted k for better spacing
			print(f"Position:\n{pos}")
			edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
			print(f"Edge weights:\n{edge_weights}")
			nx.draw_networkx_edges(
				G=G, 
				pos=pos,
				width=[w * 0.1 for w in edge_weights],
				alpha=0.65,
				edge_color='#303030b7'
			)
			nx.draw_networkx_nodes(
				G, 
				pos,
				node_size=100,
				node_color='#1070b1',
				alpha=0.9,
			)
			nx.draw_networkx_labels(
				G, 
				pos,
				font_size=5,
				alpha=0.95,
				verticalalignment='baseline',
				horizontalalignment='left'
			)
			plt.title(f'Phrase Co-Occurrence Network for Topic {label} [before merging] ({len(G.nodes())} nodes, {len(G.edges())} edges)')
			plt.axis('off')
			plt.savefig(os.path.join(dataset_dir, f'cooccurrence_network_topic_{label}_before_merging.png'), bbox_inches='tight', dpi=300)
			plt.close()

	# Visualization 5: Top-K Phrases Bar Plot for Each Topic [before merging]
	if enable_visualizations:
		for label, topic_phrases in enumerate(topics_before_merging):
			if not topic_phrases:
				# print(f"Skipping Topic {label}: No phrases available.")
				continue
			counter = cluster_phrases[label]
			# print(f"Topic {label}: {len(counter)} phrases, Selecting Top-{len(topic_phrases)}")
			phrase_freq = {phrase: counter[phrase] for phrase in topic_phrases if phrase in counter}
			top_k_phrases_per_topic_before_merging = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
			if not top_k_phrases_per_topic_before_merging:
				continue
			phrases, frequencies = zip(*top_k_phrases_per_topic_before_merging)
			plt.figure(figsize=(14, 6))
			sns.barplot(x=frequencies, y=phrases, palette='Blues_r')
			plt.title(f'Top {len(top_k_phrases_per_topic_before_merging)} Phrases {len(phrase_freq)} in Topic {label} [before merging]')
			plt.xlabel('Frequency')
			plt.ylabel('Phrase')
			plt.savefig(os.path.join(dataset_dir, f'topK_phrases_topic_{label}_before_merging.png'), bbox_inches='tight')
			plt.close()

	# Visualization 6: Topic Similarity Heatmap (before merging)
	if enable_visualizations:
		plt.figure(figsize=(17, 12))
		sns.heatmap(
			data=similarity_matrix, 
			# annot=True, 
			cmap='YlOrRd', 
			vmin=0, 
			vmax=1, 
			square=True,
		)
		plt.title(f'Topic Similarity Matrix (Cosine Similarity) [before merging]')
		plt.xlabel('Topic ID')
		plt.ylabel('Topic ID')
		plt.savefig(os.path.join(dataset_dir, f'topic_similarity_heatmap_before_merging.png'), bbox_inches='tight')
		plt.close()
	
	# Visualization 7: Dendrogram of Topic Similarities (before merging)
	if enable_visualizations:
		sim_values = similarity_matrix[np.triu_indices(len(topics_before_merging), k=1)]
		if sim_values.size > 0:
			mean_sim = np.mean(sim_values)
			min_sim = np.min(sim_values)
			max_sim = np.max(sim_values)
			print(f"Similarity matrix stats: Mean={mean_sim:.3f}, Min={min_sim:.3f}, Max={max_sim:.3f}")
			merge_threshold = np.percentile(sim_values, 75) + 0.10
			print(f"Dynamic merge threshold (75th percentile): {merge_threshold:.4f}")
			# Convert similarity → distance
			dist_matrix = 1.0 - similarity_matrix
			dist_matrix = np.clip(dist_matrix, 0, 2)  # Ensure all distances are in [0, 2]
			# Zero out diagonal to make it a valid distance matrix
			np.fill_diagonal(dist_matrix, 0.0)
			from scipy.spatial.distance import squareform
			condensed_dist = squareform(dist_matrix)  # Converts to condensed form for linkage
			from scipy.cluster.hierarchy import linkage, dendrogram
			plt.figure(figsize=(17, 10))
			linkage_matrix = linkage(condensed_dist, method='average')
			dendrogram(
				linkage_matrix,
				labels=[f'Topic {i}' for i in range(len(topics_before_merging))],
				color_threshold=1 - merge_threshold
			)
			plt.title('Dendrogram of Topic Similarities [before merging]')
			plt.xlabel('Topics')
			plt.ylabel('Distance (1 - Cosine Similarity)')
			plt.axhline(y=1 - merge_threshold, color='red', linestyle='--', label=f'Merge Threshold ({merge_threshold:.4f})')
			plt.legend()
			plt.xticks(rotation=90, fontsize=8)
			plt.savefig(os.path.join(dataset_dir, f'similarity_dendrogram_before_merging_thresh_{merge_threshold:.4f}.png'), bbox_inches='tight')
			plt.close()
		else:
			print("Similarity matrix is empty.")

	# # Visualization 8: UMAP with Top Phrases
	# if enable_visualizations:
	# 	# Verify outliers definition (HDBSCAN noise points)
	# 	outliers = labels == -1  # Noise points from HDBSCAN
	# 	print(f"Noise points (outliers) in UMAP plot: {np.sum(outliers)}/{len(texts)} texts [{np.sum(outliers) / len(texts) * 100:.2f}%]")
		
	# 	# Get unique clusters (excluding noise)
	# 	unique_clusters = np.unique(labels[~outliers])
	# 	print(f">> {len(unique_clusters)} Unique Clusters (excluding noise) [{np.sum(outliers) / len(texts) * 100:.2f}% noise]")
	# 	if len(unique_clusters) > 0:
	# 		# Calculate centroids in 2D UMAP space
	# 		centroids = np.zeros((n_clusters, 2))
	# 		for i in range(n_clusters):
	# 			cluster_points = emb_umap[labels == i]
	# 			if len(cluster_points) > 0:
	# 				centroids[i] = np.mean(cluster_points, axis=0)
	# 		print(f"Centroids shape: {centroids.shape}")
			
	# 		# Assign outliers to nearest cluster based on distance
	# 		outlier_assignments = np.full(emb_umap.shape[0], -1)
	# 		if np.sum(outliers) > 0:
	# 			# Compute distances from outlier points to centroids
	# 			outlier_indices = np.where(outliers)[0]
	# 			outlier_points = emb_umap[outlier_indices]
	# 			distances = np.linalg.norm(outlier_points[:, np.newaxis] - centroids, axis=2)
	# 			# Assign each outlier to the nearest cluster
	# 			nearest_clusters = unique_clusters[np.argmin(distances, axis=1)]
	# 			outlier_assignments[outlier_indices] = nearest_clusters
			
	# 		plt.figure(figsize=(18, 10))
	# 		# Map cluster labels to colors from the 'tab20' palette
	# 		tab20_cmap = plt.cm.get_cmap('tab20')
	# 		cluster_colors = tab20_cmap(np.linspace(0, 1, len(unique_clusters)))
	# 		# Create a mapping of cluster labels to colors
	# 		cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
			
	# 		# Plot inliers as empty circles with edge colors matching their cluster
	# 		for cluster in unique_clusters:
	# 			cluster_mask = labels == cluster
	# 			plt.scatter(
	# 				emb_umap[cluster_mask, 0],
	# 				emb_umap[cluster_mask, 1],
	# 				facecolors='none',
	# 				edgecolors=cluster_color_map[cluster],
	# 				marker='o',
	# 				s=30,
	# 				linewidths=1.1,
	# 				alpha=0.98,
	# 				label=None,
	# 				zorder=2,
	# 			)
			
	# 		# Plot outliers with same color as nearest cluster, less transparency
	# 		if np.sum(outliers) > 0:
	# 			plt.scatter(
	# 				emb_umap[outliers, 0],
	# 				emb_umap[outliers, 1],
	# 				# facecolors='none',
	# 				facecolors=[cluster_color_map[cluster] for cluster in outlier_assignments[outliers]],
	# 				marker='^',
	# 				s=15,
	# 				linewidths=1.0,
	# 				alpha=0.7,
	# 				label=None,
	# 				zorder=1,
	# 			)
	# 		# Plot centroids with colors matching their clusters
	# 		for i, cluster in enumerate(unique_clusters):
	# 			plt.scatter(
	# 				centroids[cluster, 0],
	# 				centroids[cluster, 1],
	# 				c=[cluster_color_map[cluster]],
	# 				marker='x',
	# 				s=300,
	# 				linewidths=3.5,
	# 				alpha=0.75,
	# 				zorder=3,
	# 			)
			
	# 		plt.title(f'2D UMAP Visualization of Text Embeddings with Top Phrases for {len(unique_clusters)} Clusters')
	# 		plt.xlabel('UMAP 1')
	# 		plt.ylabel('UMAP 2')
	# 		ax = plt.gca()
	# 		if ax.legend_ is not None:
	# 			ax.legend_.remove()
	# 		plt.savefig(os.path.join(dataset_dir, 'umap_cluster_visualization_with_phrases.png'), bbox_inches='tight')
	# 		plt.close()
	# 	else:
	# 		print("No unique clusters found, skipping UMAP visualization with top phrases...")

	# Visualization 9: Cluster Size vs. Term Count Plot
	if enable_visualizations:
		print("Cluster Size vs. Term Count Plot")
		print(f"Cluster Text Counts(before merging) {len(cluster_text_counts)}: {cluster_text_counts}")
		print(f"Term Counts per Cluster(before merging): {len(term_counts_per_cluster)}: {term_counts_per_cluster}")
		# Align clusters with valid phrases
		valid_clusters = [i for i in range(n_clusters) if i in cluster_text_counts and i < len(term_counts_per_cluster)]
		if valid_clusters:
			plt.figure(figsize=(19, 13))
			sns.scatterplot(
				x=[cluster_text_counts[i] for i in valid_clusters],
				y=[term_counts_per_cluster[i] for i in valid_clusters],
				hue=[i for i in valid_clusters],
				palette='tab20',
				size=[term_counts_per_cluster[i] for i in valid_clusters],
				sizes=(50, 500),
				legend=False,
			)
			for i in valid_clusters:
				plt.text(
					cluster_text_counts[i], 
					term_counts_per_cluster[i] + 5,
					f'Topic {i}',
					ha='center', 
					va='bottom',
					fontsize=10,
				)
			plt.title('Cluster Size vs. Number of Unique Terms')
			plt.xlabel('Number of Documents in Cluster')
			plt.ylabel('Number of Unique Terms')
			ax = plt.gca()
			if ax.legend_ is not None:
				ax.legend_.remove()
			# plt.legend(title='Cluster', bbox_to_anchor=(1.01, 1), loc='upper left')
			plt.savefig(os.path.join(dataset_dir, 'cluster_size_vs_term_count.png'), bbox_inches='tight')
			plt.close()
		else:
			print("No valid clusters found, skipping Cluster Size vs. Term Count Plot...")

	# Visualization 10: Interactive Topic Similarity Network
	if enable_visualizations:
		G = nx.Graph()
		for i in range(len(topics_before_merging)):
				G.add_node(i, label=f'Topic {i}')
		edge_weights = []
		for i in range(len(topics_before_merging)):
				for j in range(i + 1, len(topics_before_merging)):
						sim = similarity_matrix[i, j]
						if sim > 0.5:
								G.add_edge(i, j, weight=sim)
								edge_weights.append(sim)
		pos = nx.spring_layout(G)
		edge_x = []
		edge_y = []
		edge_text = []
		for edge in G.edges():
				x0, y0 = pos[edge[0]]
				x1, y1 = pos[edge[1]]
				edge_x.extend([x0, x1, None])
				edge_y.extend([y0, y1, None])
				weight = G[edge[0]][edge[1]]['weight']
				edge_text.extend([f'Similarity: {weight:.2f}', None, None])
		node_x = [pos[i][0] for i in G.nodes()]
		node_y = [pos[i][1] for i in G.nodes()]
		node_text = [f'Topic {i}' for i in G.nodes()]
		fig = go.Figure()
		fig.add_trace(go.Scatter(
				x=edge_x, y=edge_y,
				mode='lines',
				line=dict(width=2, color='gray'),
				hoverinfo='text',
				text=edge_text,
				showlegend=False
		))
		fig.add_trace(go.Scatter(
				x=node_x, y=node_y,
				mode='markers+text',
				text=node_text,
				textposition='top center',
				marker=dict(size=20, color='lightblue'),
				hoverinfo='text',
				name='Topics'
		))
		fig.update_layout(
				title='Interactive Topic Similarity Network (Edges for Similarity > 0.5)',
				showlegend=True,
				hovermode='closest',
				margin=dict(b=20, l=5, r=5, t=40),
				xaxis=dict(showgrid=False, zeroline=False),
				yaxis=dict(showgrid=False, zeroline=False)
		)
		fig.write_html(os.path.join(dataset_dir, 'topic_similarity_network_interactive.html'))

	# Merge similar topics
	print(f"\n>> Merging similar topics with dynamic merging threshold {merge_threshold:.4f}")
	merged_topics = []
	used_indices = set()
	min_topics = max(1, len(topics_before_merging) // 2)
	max_merges = len(topics_before_merging) - min_topics
	merge_count = 0
	for i in range(len(topics_before_merging)):
		if i in used_indices:
			continue
		merged_words = set(topics_before_merging[i])
		merged_indices = {i}
		used_indices.add(i)
		for j in range(len(topics_before_merging)):
			if j in used_indices or i == j or merge_count >= max_merges:
				continue
			if similarity_matrix[i, j] > merge_threshold:
				merged_words.update(topics_before_merging[j])
				merged_indices.add(j)
				used_indices.add(j)
				merge_count += 1
		
		# Aggregate frequencies and prioritize diversity
		counter = Counter()
		for topic_idx in merged_indices:
			for phrase in topics_before_merging[topic_idx]:
				for orig_counter in cluster_phrases.values():
					if phrase in orig_counter:
						counter[phrase] += orig_counter[phrase]
		
		# Score phrases with diversity bonus
		phrase_scores = []
		seen_words = set()
		for phrase, count in counter.items():
			words = set(phrase.split())
			diversity_bonus = sum(1 for word in words if word not in seen_words)
			score = count * (1 + 0.2 * diversity_bonus)
			phrase_scores.append((phrase, count, score))
			seen_words.update(words)
		phrase_scores.sort(key=lambda x: x[2], reverse=True)
		print(f"Topic {i} contains {len(counter)} phrases before merging")
		sorted_phrases = [
			phrase 
			for phrase, _, _ in phrase_scores[:top_k_words]
		]
		print(f"\tTopic {i} contains {len(sorted_phrases)} phrases after merging")
		if sorted_phrases:
			merged_topics.append(sorted_phrases)
	print(f"Topics Reduced from {len(topics_before_merging)} to {len(merged_topics)} topics after merging with meriging threshold {merge_threshold}")

	# Visualization 11: Topic Diversity Plot (after merging)
	if enable_visualizations:
		unique_terms_per_topic = [len(set(topic)) for topic in merged_topics]
		plt.figure(figsize=(17, 7))
		plt.bar(range(len(merged_topics)), unique_terms_per_topic, color='#3785e6')
		plt.title('Number of Unique Terms per Topic (after merging)')
		plt.xlabel('Topic ID')
		plt.ylabel('Number of Unique Terms')
		plt.xticks(range(len(merged_topics)))
		plt.savefig(os.path.join(dataset_dir, 'merged_topic_diversity.png'), bbox_inches='tight')
		plt.close()

		# Compute topic diversity (Jaccard similarity between topics)
		topic_sets = [set(topic) for topic in merged_topics]
		jaccard_matrix = np.zeros((len(merged_topics), len(merged_topics)))
		for i in range(len(merged_topics)):
			for j in range(i + 1, len(merged_topics)):
				intersection = len(topic_sets[i] & topic_sets[j])
				union = len(topic_sets[i] | topic_sets[j])
				jaccard_matrix[i, j] = intersection / union if union > 0 else 0
				jaccard_matrix[j, i] = jaccard_matrix[i, j]
		plt.figure(figsize=(15, 11))
		sns.heatmap(
			data=jaccard_matrix, 
			# annot=True, 
			cmap='YlGnBu', 
			vmin=0, 
			vmax=1, 
			square=True,
		)
		plt.title(f'Merged Topic Diversity (Jaccard Similarity Between Topics) meriging threshold {merge_threshold}')
		plt.xlabel('Topic ID')
		plt.ylabel('Topic ID')
		plt.savefig(os.path.join(dataset_dir, 'merged_topic_diversity_jaccard_similarity.png'), bbox_inches='tight')
		plt.close()

	# Visualization 12: Topic Distribution Comparison: Before vs. After Merging
	if enable_visualizations:
		plt.figure(figsize=(19, 10))
		bars_before = plt.bar(
			[i - 0.1 for i in range(n_clusters)],
			[cluster_text_counts[i] for i in range(n_clusters) if i in cluster_text_counts],
			width=0.2,
			color='#3785e6',
			label=f'Before Merging: {sum(cluster_text_counts[i] for i in range(n_clusters))} in {n_clusters} clusters'
		)
		bars_after = plt.bar(
			[i + 0.1 for i in range(len(merged_topics))],
			[len(topic) for topic in merged_topics],
			width=0.2,
			color='#e45151',
			label=f'After Merging (Terms): {sum(len(topic) for topic in merged_topics)} in {len(merged_topics)} clusters'
		)
		plt.title('Cluster Distribution: Before vs. After Merging')
		plt.xlabel('Cluster/Topic')
		plt.xticks(range(max(n_clusters, len(merged_topics))), labels=range(max(n_clusters, len(merged_topics))), fontsize=10, va='center', ha='center', rotation=90)
		plt.ylabel('Count')
		plt.legend()
		plt.savefig(os.path.join(dataset_dir, f'topic_distribution_original_vs_merged_thresh_{merge_threshold:.4f}.png'), bbox_inches='tight')
		plt.close()
	
	# Visualization 13: Noise Analysis Plot
	if enable_visualizations:
		noise_texts = [texts[i] for i, label in enumerate(labels) if label == -1]
		noise_lengths = [len(text.split()) for text in noise_texts]
		noise_stopword_ratios = [sum(1 for word in text.split() if word in CUSTOM_STOPWORDS) / max(1, len(text.split())) for text in noise_texts]
		
		plt.figure(figsize=(18, 10))
		plt.subplot(1, 2, 1)
		plt.hist(noise_lengths, bins=50, color='salmon', edgecolor='black')
		plt.title('Text Length Distribution of Noise Points')
		plt.xlabel('Text Length (Words)')
		plt.ylabel('Number of Texts')
		plt.subplot(1, 2, 2)
		plt.hist(noise_stopword_ratios, bins=50, color='lightgreen', edgecolor='black')
		plt.title('Stopword Ratio Distribution of Noise Points')
		plt.xlabel('Stopword Ratio')
		plt.ylabel('Number of Texts')
		plt.tight_layout()
		plt.savefig(os.path.join(dataset_dir, 'noise_analysis.png'), bbox_inches='tight')
		plt.close()

	# Visualization 14: Phrase Co-Occurrence Network for Each Topic [after merging]
	if enable_visualizations:
		print(f"Generating co-occurrence networks for {len(merged_topics)} topics [after merging]...")
		# Map merged topic indices to original cluster indices
		cluster_to_merged = {}
		current_merged_idx = 0
		used_indices = set()
		for i in range(len(topics_before_merging)):
			if i in used_indices:
				continue
			merged_indices = {i}
			used_indices.add(i)
			for j in range(len(topics_before_merging)):
				if j in used_indices or i == j:
					continue
				if similarity_matrix[i, j] > merge_threshold:
					merged_indices.add(j)
					used_indices.add(j)
			for orig_idx in merged_indices:
				cluster_to_merged[orig_idx] = current_merged_idx
			current_merged_idx += 1
		for merged_idx, topic_phrases in enumerate(merged_topics):
			if not topic_phrases:
					# print(f"Skipping Merged Topic {merged_idx}: No phrases available.")
					continue
			# Select top 10 phrases by frequency to reduce clutter
			counter = Counter()
			original_clusters = [orig_idx for orig_idx, m_idx in cluster_to_merged.items() if m_idx == merged_idx]
			for orig_idx in original_clusters:
					for phrase in cluster_phrases[orig_idx]:
							counter[phrase] += cluster_phrases[orig_idx][phrase]
			phrase_freq = {phrase: counter[phrase] for phrase in topic_phrases if phrase in counter}
			top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
			top_phrases = [phrase for phrase, _ in top_phrases]
			if not top_phrases:
					# print(f"Skipping Merged Topic {merged_idx}: No phrases after filtering.")
					continue
			# Collect texts from all original clusters that were merged into this topic
			cluster_texts = [
					texts[i] for i, l in enumerate(labels)
					if l in original_clusters and is_english(texts[i], ft_model)
			]
			if not cluster_texts:
					print(f"Skipping Merged Topic {merged_idx}: No valid texts for co-occurrence.")
					continue
			phrase_set = set(top_phrases)
			cooc_matrix = defaultdict(int)
			for text in cluster_texts:
				phrases = kw_model.extract_keywords(
					text,
					keyphrase_ngram_range=(1, 3),
					stop_words="english",
					top_n=15 if len(text.split()) > 100 else 5,
					diversity=0.7,
				)
				valid_phrases = []
				seen_phrases = set()
				for phrase, _ in phrases:
					words = phrase.split()
					stopword_count = sum(1 for w in words if w in CUSTOM_STOPWORDS)
					if stopword_count / len(words) > 0.6 or len(words) < 2:
						continue
					normalized = " ".join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])
					if len(normalized.split()) >= 2 and normalized not in seen_phrases:
						valid_phrases.append(normalized)
						seen_phrases.add(normalized)
				text_phrases = set(valid_phrases).intersection(phrase_set)
				for p1 in text_phrases:
					for p2 in text_phrases:
						if p1 < p2:
							cooc_matrix[(p1, p2)] += 1
			G = nx.Graph()
			for (p1, p2), count in cooc_matrix.items():
				if count >= 2: # An edge exists between two phrases if they appear together in the same text at least twice
					G.add_edge(p1, p2, weight=count)
			for phrase in top_phrases:
				if phrase not in G:
					G.add_node(phrase)
			plt.figure(figsize=(12, 8))
			pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjusted k for better spacing
			edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
			nx.draw_networkx_edges(
				G=G, 
				pos=pos,
				width=[w * 0.1 for w in edge_weights],  # Scale edge width
				alpha=0.6,
				edge_color='#474646'
			)
			nx.draw_networkx_nodes(
				G=G, 
				pos=pos,
				node_size=150,
				node_color='#df9100',
				alpha=0.9
			)
			nx.draw_networkx_labels(
				G=G, 
				pos=pos,
				font_size=5,
				alpha=0.95,
				verticalalignment='center',
				horizontalalignment='center'
			)
			plt.title(f'Phrase Co-Occurrence Network for Merged Topic {merged_idx} ({len(G.nodes())} nodes, {len(G.edges())} edges)')
			plt.axis('off')
			plt.savefig(os.path.join(dataset_dir, f'cooccurrence_network_merged_topic_{merged_idx}.png'), bbox_inches='tight', dpi=300)
			plt.close()

	flat_topics = set(word for topic in merged_topics for word in topic)
	print(f"Extracted {len(flat_topics)} unique topic terms after merging")
	return merged_topics, flat_topics


def get_visual_based_annotation_old(
		csv_file: str,
		vlm_model_name: str,
		batch_size: int,
		device: str,
		num_workers: int,
		verbose: bool,
		metadata_fpth: str,
		topk: int = 3,
	):
	if verbose:
		print(f"Semi-Supervised label extraction from image data (using VLM) batch_size: {batch_size}".center(160, "-"))
	
	visual_based_annotation_start_time = time.time()
	
	# Load categories
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	texts = [f"This is a photo of {lbl}." for lbl in candidate_labels]
	
	gpu_name = torch.cuda.get_device_name(device)
	total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB
	available_gpu_memory = torch.cuda.mem_get_info()[0] / 1024**3 # GB
	print(f"Total GPU memory: {total_gpu_memory:.2f} GB ({gpu_name})")
	print(f"Available GPU memory: {available_gpu_memory:.2f} GB ({gpu_name})")

	# Setup model
	model = AutoModel.from_pretrained(
		pretrained_model_name_or_path=vlm_model_name,
		torch_dtype=torch.float16 if available_gpu_memory < 7 else torch.float32,
		device_map=device,
	).eval()
	if available_gpu_memory > 7:
		model = torch.compile(model, mode="reduce-overhead")
	print(model.parameters().__next__().dtype)
	processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=vlm_model_name)
	
	text_inputs = processor(
		text=texts,
		padding="max_length",
		max_length=64,
		return_tensors="pt",
	).to(device)
	
	with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
		text_embeddings = model.get_text_features(**text_inputs)
		text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
	
	# Load dataframe
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'enriched_document_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	}
	if verbose:
		print(f"Loading metadata from {csv_file}...")
	df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")

	img_paths = df['img_path'].tolist()

	combined_labels = [[] for _ in range(len(img_paths))]

	img_path_batches = [img_paths[i:i+batch_size] for i in range(0, len(img_paths), batch_size)]

	for batch_idx, batch in enumerate(tqdm(img_path_batches, desc="Processing batched images")):
		batch_start_idx = batch_idx * batch_size
		
		images = []
		valid_global_indices = []
		
		# Load images with proper index tracking
		for local_i, pth in enumerate(batch):
			global_idx = batch_start_idx + local_i
			
			try:
				if os.path.exists(pth):
					image = Image.open(pth).convert("RGB")
					images.append(image)
					valid_global_indices.append(global_idx)
				else:
					combined_labels[global_idx] = []
			except Exception as e:
				print(f"ERROR: failed to load image from {pth} => {e}")
				combined_labels[global_idx] = []
		
		# Skip if no valid images in batch
		if not images:
			continue
		
		try:
			# Process batch
			image_inputs = processor(
				images=images,
				padding="max_length",
				max_num_patches=4096,
				return_tensors="pt",
			).to(device)
			
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				image_embeddings = model.get_image_features(**image_inputs)
				image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
				similarities = image_embeddings @ text_embeddings.T
			
			for processed_idx, global_idx in enumerate(valid_global_indices):
				topk_probs, topk_indices = similarities[processed_idx].topk(topk)
				topk_labels = [candidate_labels[idx] for idx in topk_indices]
				combined_labels[global_idx] = topk_labels
			
			del image_inputs, image_embeddings, similarities
		except Exception as e:
			print(f"ERROR: failed to process batch {batch_start_idx}-{batch_start_idx + len(batch)}: {e}")
			# Assign empty results for failed batch
			for global_idx in valid_global_indices:
				if not combined_labels[global_idx]:  # Only if not already assigned
					combined_labels[global_idx] = []
		
		# Memory cleanup every few batches
		if batch_idx % 10 == 0:
			torch.cuda.empty_cache()

	df['visual_based_labels'] = combined_labels
	df.to_csv(metadata_fpth, index=False)
	print(f"Processed {len(img_paths)} images, generated {sum(1 for labels in combined_labels if labels)} valid results")
	print(f"Visual-based annotation Elapsed time: {time.time() - visual_based_annotation_start_time:.2f} sec".center(160, " "))
	return combined_labels


	# Visualization 0: Raw Embeddings into 2D for better debugging
	print(f"Reducing embeddings: {embeddings.shape} to 2D for visualization using UMAP")
	umap_reducer = umap.UMAP(
		n_neighbors=15,
		min_dist=0.1,
		densmap=True,
		spread=1.0,
		n_components=2, 
		random_state=42, 
		metric='cosine',
	)
	emb_umap = umap_reducer.fit_transform(embeddings)
	if enable_visualizations:
		# Visualize embeddings with UMAP
		print(f"Reducing embeddings: {embeddings.shape} to 2D for visualization using UMAP")
		plt.figure(figsize=(18, 10))
		plt.scatter(emb_umap[:, 0], emb_umap[:, 1], s=25, c='#f1f8ff', edgecolors='#0078e9', alpha=0.8)
		plt.title(f"UMAP Visualization of Embeddings ({dataset_size} Texts)")
		plt.xlabel("UMAP Dimension 1")
		plt.ylabel("UMAP Dimension 2")
		plt.savefig(os.path.join(dataset_dir, f'umap_raw_embeddings_{embeddings.shape[0]}_x_{embeddings.shape[1]}.png'), bbox_inches='tight')
		print(f"UMAP visualization saved to {os.path.join(dataset_dir, f'umap_raw_embeddings_{embeddings.shape[0]}_x_{embeddings.shape[1]}.png')}")


############################################################################################################################
def evaluate_retrieval_performance(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		device: str,
		topK_values: List[int],
	):
	start_time = time.time()
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f">> Evaluating {model_name} {model_arch} Retrieval Performance [{dataset_name}]: {topK_values}...")
	model.eval()  # dropout is disabled, ensuring deterministic outputs
	image_embeddings = []
	image_labels = []

	try:
		class_names = validation_loader.dataset.dataset.classes
	except:
		class_names = validation_loader.dataset.unique_labels
	
	n_classes = len(class_names)
	torch.cuda.empty_cache() # Clear GPU memory cache

	with torch.no_grad():
		text_inputs = clip.tokenize(texts=class_names).to(device, non_blocking=True)
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			class_text_embeddings = model.encode_text(text_inputs)
		class_text_embeddings = class_text_embeddings / class_text_embeddings.norm(dim=-1, keepdim=True)
		for bidx, (images, _, class_indices) in enumerate(validation_loader):
			images = images.to(device, non_blocking=True)
			class_indices = class_indices.to(device, non_blocking=True)
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				image_embeds = model.encode_image(images)
			image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
			image_embeddings.append(image_embeds.cpu())
			image_labels.extend(class_indices.cpu())

	# Aggregate and normalize embeddings
	image_embeddings = torch.cat(image_embeddings, dim=0)
	image_labels = torch.tensor(image_labels)
	class_text_embeddings = class_text_embeddings.cpu()
	similarity_matrix = image_embeddings @ class_text_embeddings.T

	# print("Similarity matrix stats:")
	# print(
	# 		type(similarity_matrix),
	# 		similarity_matrix.shape,
	# 		similarity_matrix.dtype,
	# 		similarity_matrix.min(),
	# 		similarity_matrix.max(),
	# 		similarity_matrix.mean(),
	# 		similarity_matrix.std(),
	# )
	# print(similarity_matrix[:10, :10])  # ensure values are reasonable (e.g., -1 to 1).

	image_to_text_metrics = get_retrieval_metrics(
		similarity_matrix=similarity_matrix,
		query_labels=image_labels,
		candidate_labels=torch.arange(n_classes),
		topK_values=topK_values,
		mode="Image-to-Text",
		class_counts=None,  # No class counts for Image-to-Text
		max_k=n_classes,  # Pass max_k for Image-to-Text to limit K to the number of classes
	)

	text_to_image_metrics = get_retrieval_metrics(
		similarity_matrix=class_text_embeddings @ image_embeddings.T,
		query_labels=torch.arange(n_classes),
		candidate_labels=image_labels,
		topK_values=topK_values,
		mode="Text-to-Image",
		class_counts=torch.bincount(image_labels),  # Count number of occurrences of each value in array of non-negative ints.
		max_k=None,  # No limit on K for Text-to-Image
	)
	metrics = {
		"img2txt": image_to_text_metrics,
		"txt2img": text_to_image_metrics,
	} 
	print(f"Elapsed_t: {time.time()-start_time:.1f} sec".center(150, "-"))

	return metrics

def get_retrieval_metrics(
		similarity_matrix: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor,
		topK_values: List[int],
		mode: str = "Image-to-Text",
		class_counts: torch.Tensor = None,
		max_k: int = None, # limit K values (None for no limit)
	):

	num_queries, num_candidates = similarity_matrix.shape
	assert num_queries == len(query_labels), "Number of queries must match labels"
	assert query_labels.device == candidate_labels.device, "query_labels and candidate_labels must be on the same device"
	num_classes = len(torch.unique(candidate_labels))

	if max_k is not None:
		valid_K_values = [K for K in topK_values if K <= max_k]
	else:
		valid_K_values = topK_values  # No limit on K values

	if len(valid_K_values) < len(topK_values):
		print(f"\t<!> Warning: K values: ({set(topK_values) - set(valid_K_values)}) exceed the number of classes ({num_classes}) => ignored!")

	metrics = {
		"mP": {},
		"mAP": {},
		"Recall": {},
	}

	for K in valid_K_values:
		top_k_indices = torch.argsort(-similarity_matrix, dim=1)[:, :K].cpu()
		precision, recall, ap = [], [], []
		for i in range(num_queries):
			true_label = query_labels[i]
			retrieved_labels = candidate_labels[top_k_indices[i]]
			correct = (retrieved_labels == true_label).sum().item()
			
			# 1. Precision @ K
			precision.append(correct / K)

			# 2. Compute Recall@K with division by zero protection
			if mode == "Image-to-Text":
				relevant_count = 1  # Single relevant item per query [single label per image]
			else:
				relevant_count = (
					class_counts[true_label].item()
					if class_counts is not None
					else 0
				)
			if relevant_count == 0:
				recall.append(0.0)
			else:
				recall.append(correct / relevant_count)
			
			# 3. Compute AP@K with proper normalization
			relevant_positions = torch.where(retrieved_labels == true_label)[0]
			p_at = []
			cumulative_correct = 0
			for pos in relevant_positions:
				if pos < K:  # Only consider positions within top-K
					cumulative_correct += 1
					precision_at_rank = cumulative_correct / (pos + 1)  # pos is 0-based
					p_at.append(precision_at_rank)
			# Determine normalization factor
			if mode == "Image-to-Text":
				R = 1 # Always 1 relevant item for image-to-text
			else:
				R = (
					class_counts[true_label].item()
					if class_counts is not None
					else 0
				)
			# Handle queries with no relevant items or no retrieved items
			if R == 0 or len(p_at) == 0:
				ap.append(0.0)
			else:
				ap.append(sum(p_at) / min(R, K))  # Normalize by min(R, K)
		# Store metrics for this K
		metrics["mP"][str(K)] = torch.tensor(precision).mean().item()
		metrics["mAP"][str(K)] = torch.tensor(ap).mean().item()
		metrics["Recall"][str(K)] = torch.tensor(recall).mean().item()
	return metrics

def get_in_batch_validation_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str = "cuda",
		topK_values: List[int] = [1, 3, 5],
	):
	start_time = time.time()
	torch.cuda.empty_cache()

	free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
	available_mem = free_mem * 0.95  # Use 80% of available memory

	embed_dim = model.embed_dim if hasattr(model, 'embed_dim') else 512  # Standard CLIP embedding dimension
	print(f"{hasattr(model, 'embed_dim')} | embed_dim: {embed_dim}")
	mem_per_sample = embed_dim * 4 * 2 / (1024 ** 3)  # Memory needed per sample in GB
	max_samples_in_memory = int(available_mem / mem_per_sample)
	chunk_size = max(32, min(max_samples_in_memory, 1024))
	
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f">> Evaluating {model_name} {model_arch} [in-batch Loss & Accuracy] [{dataset_name}]: {topK_values}...")
	print(f"   Using memory-efficient chunking with chunk size: {chunk_size} (based on {free_mem:.2f}GB free GPU memory)")
	
	model.eval()
	total_loss = 0
	total_img2txt_correct = 0
	total_txt2img_correct = 0
	num_batches = len(validation_loader)
	total_samples = len(validation_loader.dataset)
	
	try:
		class_names = validation_loader.dataset.dataset.classes
	except:
		class_names = validation_loader.dataset.unique_labels
	
	num_classes = len(class_names)
	
	valid_img2txt_k_values = [K for K in topK_values if K <= num_classes]
	if len(valid_img2txt_k_values) < len(topK_values):
		print(f"\t<!> Warning: K values ({set(topK_values) - set(valid_img2txt_k_values)}) exceed the number of classes ({num_classes}) for Image-to-Text. => ignored.")
	
	valid_txt2img_k_values = topK_values
	img2txt_topk_accuracy = {k: 0 for k in valid_img2txt_k_values}
	txt2img_topk_accuracy = {k: 0 for k in valid_txt2img_k_values}
	cosine_similarities = []
	
	# Store all embeddings and labels for efficient processing
	all_image_embeddings = []
	all_text_embeddings = []
	all_labels = []
	
	# First pass: compute loss and collect embeddings
	with torch.no_grad():
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
				batch_size = images.size(0)
				
				# Process data in chunks to save memory
				for chunk_start in range(0, batch_size, chunk_size):
						chunk_end = min(chunk_start + chunk_size, batch_size)
						chunk_images = images[chunk_start:chunk_end].to(device, non_blocking=True)
						chunk_tokenized_labels = tokenized_labels[chunk_start:chunk_end].to(device, non_blocking=True)
						chunk_labels_indices = labels_indices[chunk_start:chunk_end].to(device, non_blocking=True)
						chunk_size_actual = chunk_images.size(0)
						
						# Forward pass to get logits
						with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
							logits_per_image, logits_per_text = model(chunk_images, chunk_tokenized_labels)
							# Ground Truth
							ground_truth = torch.arange(start=0, end=chunk_size_actual, dtype=torch.long, device=device)
						
							# Validation Loss: Average of both losses
							loss_img = criterion(logits_per_image, ground_truth)
							loss_txt = criterion(logits_per_text, ground_truth)
							batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
							total_loss += batch_loss * (chunk_size_actual / batch_size)  # Weight by portion of batch
						
						# Map in-batch indices to actual labels
						batch_labels = chunk_labels_indices  # [chunk_size], actual class indices
						
						# Top-1 Accuracy for Image-to-Text
						pred_lbl_per_img_idxs = torch.argmax(logits_per_image, dim=1)  # [chunk_size]
						img2txt_correct = 0
						for i in range(chunk_size_actual):
								pred_idx = pred_lbl_per_img_idxs[i].item()
								true_label = batch_labels[i].item()
								pred_label = batch_labels[pred_idx].item()
								if pred_label == true_label:
										img2txt_correct += 1
						total_img2txt_correct += img2txt_correct
						
						# Top-1 Accuracy for Text-to-Image
						pred_img_per_lbl_idxs = torch.argmax(logits_per_text, dim=1)  # [chunk_size]
						txt2img_correct = 0
						for i in range(chunk_size_actual):
								pred_idx = pred_img_per_lbl_idxs[i].item()
								true_label = batch_labels[i].item()
								pred_label = batch_labels[pred_idx].item()
								if pred_label == true_label:
										txt2img_correct += 1
						total_txt2img_correct += txt2img_correct
						
						# Top-K Accuracy for Image-to-Text
						for k in valid_img2txt_k_values:
								effective_k = min(k, chunk_size_actual)
								topk_predicted_labels_values, topk_predicted_labels_idxs = torch.topk(logits_per_image, k=effective_k, dim=1)
								correct = 0
								for i in range(chunk_size_actual):
										topk_indices = topk_predicted_labels_idxs[i]  # [k]
										true_label = batch_labels[i].item()
										for idx in topk_indices:
												pred_label = batch_labels[idx].item()
												if pred_label == true_label:
														correct += 1
														break
								img2txt_topk_accuracy[k] += correct
						
						# Top-K Accuracy for Text-to-Image
						for k in valid_txt2img_k_values:
								effective_k = min(k, chunk_size_actual)
								topk_predicted_images_values, topk_predicted_images_idxs = torch.topk(logits_per_text, k=effective_k, dim=1)
								correct = 0
								for i in range(chunk_size_actual):
										topk_indices = topk_predicted_images_idxs[i]  # [k]
										true_label = batch_labels[i].item()
										for idx in topk_indices:
												pred_label = batch_labels[idx].item()
												if pred_label == true_label:
														correct += 1
														break
								txt2img_topk_accuracy[k] += correct
						
						# Compute and store embeddings efficiently
						with torch.no_grad():
								# Compute embeddings:
								with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
									image_embeddings = model.encode_image(chunk_images)
									text_embeddings = model.encode_text(chunk_tokenized_labels)
								
								image_embeddings = F.normalize(image_embeddings, dim=-1)									
								text_embeddings = F.normalize(text_embeddings, dim=-1)
								all_image_embeddings.append(image_embeddings.cpu())
								all_text_embeddings.append(text_embeddings.cpu())
								all_labels.extend(chunk_labels_indices.cpu().tolist()) # [chunk_size] store actual class indices
								
								# Compute cosine similarity
								cos_sim = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1).cpu().numpy()
								cosine_similarities.extend(cos_sim)
						
						# Clean up GPU memory
						del chunk_images, chunk_tokenized_labels, chunk_labels_indices
						del logits_per_image, logits_per_text, image_embeddings, text_embeddings
						torch.cuda.empty_cache()
				
				# if (bidx + 1) % 10 == 0 or bidx == num_batches - 1:
				# 		print(f"  Processed {bidx+1}/{num_batches} batches")
	
	# Compute average metrics
	avg_val_loss = total_loss / num_batches
	img2txt_acc = total_img2txt_correct / total_samples
	txt2img_acc = total_txt2img_correct / total_samples
	img2txt_topk_accuracy = {k: v / total_samples for k, v in img2txt_topk_accuracy.items()}
	txt2img_topk_accuracy = {k: v / total_samples for k, v in txt2img_topk_accuracy.items()}
	cosine_sim_mean = np.mean(cosine_similarities) if cosine_similarities else 0.0
	
	metrics = {
		"val_loss": float(avg_val_loss),
		"img2txt_acc": float(img2txt_acc),
		"txt2img_acc": float(txt2img_acc),
		"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_accuracy.items()},
		"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_accuracy.items()},
		"cosine_similarity": float(cosine_sim_mean),
	}
	print(f"Elapsed_t: {time.time()-start_time:.1f} sec (total samples: {total_samples})".center(150, "-"))
	return metrics

def get_full_set_validation_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str,
		print_every: int,
		topK_values: List[int],
	) -> Dict[str, float]:
	start_time = time.time()
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	
	try:
		class_names = validation_loader.dataset.dataset.classes
	except:
		class_names = validation_loader.dataset.unique_labels
	n_classes = len(class_names)
	i2t_valid_k_values = [k for k in topK_values if k <= n_classes]

	if len(i2t_valid_k_values) < len(topK_values):
		print(f"Warning: K values {set(topK_values) - set(i2t_valid_k_values)} exceed the number of classes ({n_classes}) for image-to-text retrieval")
	print(f">> Evaluating {model_name} {model_arch} [Full Loss & Accuracy] [{dataset_name}]...")
	print(f"   Image-to-Text topK: {i2t_valid_k_values}, Text-to-Image topK: {topK_values}")
	torch.cuda.empty_cache()

	free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
	available_mem = free_mem * 0.8
	embed_dim = 512
	mem_per_sample = embed_dim * 4 * 2 / (1024 ** 3)
	max_samples_in_memory = int(available_mem / mem_per_sample)
	chunk_size = max(32, min(max_samples_in_memory, 1024))
	max_img_chunk = max(32, min(max_samples_in_memory, 4096))
	print(f"Dynamic chunk size: [{chunk_size}/max:{max_img_chunk}] (based on {free_mem:.2f}GB free GPU memory)")

	model.eval()
	total_loss = 0
	num_batches = len(validation_loader)
	total_samples = len(validation_loader.dataset)
	all_image_embeds = []
	all_text_embeds = []
	cosine_similarities = []
	img2txt_mrr = []

	with torch.no_grad():
		text_inputs = clip.tokenize(class_names).to(device, non_blocking=True)
		class_text_embeddings = model.encode_text(text_inputs)
		class_text_embeddings = F.normalize(class_text_embeddings, dim=-1)
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			batch_size = images.size(0)
			logits_per_image, logits_per_text = model(images, tokenized_labels)
			image_embeds = model.encode_image(images)
			text_embeds = model.encode_text(tokenized_labels)
			image_embeds = F.normalize(image_embeds, dim=-1)
			text_embeds = F.normalize(text_embeds, dim=-1)
			ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
			loss_img = criterion(logits_per_image, ground_truth)
			loss_txt = criterion(logits_per_text, ground_truth)
			batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
			total_loss += batch_loss
			all_image_embeds.append(image_embeds.cpu())
			all_text_embeds.append(text_embeds.cpu())
			cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1).cpu().numpy()
			cosine_similarities.extend(cos_sim)
			# if bidx % print_every == 0 or bidx == num_batches - 1:
			# 	print(f"  Processed {bidx+1}/{num_batches} batches")
	avg_val_loss = total_loss / num_batches
	all_image_embeds = torch.cat(all_image_embeds, dim=0)
	all_text_embeds = torch.cat(all_text_embeds, dim=0)

	img2txt_topk_accuracy = {k: 0.0 for k in i2t_valid_k_values}
	txt2img_topk_accuracy = {k: 0.0 for k in topK_values}

	print("Computing image-to-text retrieval metrics...")
	num_chunks = (total_samples + chunk_size - 1) // chunk_size
	img2txt_top1_correct = 0
	image_labels = []

	for bidx, (_, _, labels_indices) in enumerate(validation_loader):
		batch_size = len(labels_indices)
		unique_labels = len(set(labels_indices.tolist()))
		# print(f"Batch {bidx+1}: {unique_labels} unique labels out of {batch_size} samples")
		image_labels.extend(labels_indices.tolist())

	image_labels = torch.tensor(image_labels)
	for i in range(num_chunks):
		start_idx = i * chunk_size
		end_idx = min((i + 1) * chunk_size, total_samples)
		chunk_img_embeds = all_image_embeds[start_idx:end_idx].to(device)
		chunk_labels = image_labels[start_idx:end_idx].to(device)
		similarity = chunk_img_embeds @ class_text_embeddings.to(device).T
		ranks = similarity.argsort(dim=1, descending=True)
		rr_indices = ranks.eq(chunk_labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
		rr_indices_inv = (1.0 / rr_indices).cpu().numpy()
		img2txt_mrr.extend(rr_indices_inv)
		for k in i2t_valid_k_values:
			topk_indices = similarity.topk(k, dim=1)[1]
			correct = (topk_indices == chunk_labels.unsqueeze(1)).any(dim=1).sum().item()
			img2txt_topk_accuracy[k] += correct
			if k == 1:
				img2txt_top1_correct += correct
		del chunk_img_embeds, similarity, chunk_labels
		torch.cuda.empty_cache()
		if (i + 1) % print_every == 0 or (i + 1) == num_chunks:
			print(f"  Processed {i+1}/{num_chunks} image chunks")
	img2txt_topk_accuracy = {k: v / total_samples for k, v in img2txt_topk_accuracy.items()}
	img2txt_acc = img2txt_top1_correct / total_samples

	print("Computing text-to-image retrieval metrics...")
	num_chunks = (n_classes + chunk_size - 1) // chunk_size
	txt2img_top1_correct = 0  # This will be replaced by txt2img_topk_accuracy[1]
	class_counts = torch.bincount(image_labels)
	
	print(f"Class distribution: {class_counts.tolist()}")
	for i in range(num_chunks):
		start_idx = i * chunk_size
		end_idx = min((i + 1) * chunk_size, n_classes)
		if end_idx <= start_idx:
			continue
		chunk_txt_embeds = class_text_embeddings[start_idx:end_idx].to(device)
		chunk_class_indices = torch.arange(start_idx, end_idx, device=device)
		all_topk_indices = {k: [] for k in topK_values}
		all_topk_values = {k: [] for k in topK_values}
		for j in range(0, total_samples, max_img_chunk):
			j_end = min(j + max_img_chunk, total_samples)
			img_subchunk = all_image_embeds[j:j_end].to(device)
			img_labels_subchunk = image_labels[j:j_end].to(device)
			similarity = chunk_txt_embeds @ img_subchunk.T
			for k in topK_values:
				if j == 0:
					topk_vals, topk_idxs = similarity.topk(min(k, j_end - j), dim=1)
					topk_idxs = topk_idxs + j
					all_topk_indices[k] = topk_idxs
					all_topk_values[k] = topk_vals
				else:
					current_topk_vals, current_topk_idxs = similarity.topk(min(k, j_end - j), dim=1)
					current_topk_idxs = current_topk_idxs + j
					combined_vals = torch.cat([all_topk_values[k], current_topk_vals], dim=1)
					combined_idxs = torch.cat([all_topk_indices[k], current_topk_idxs], dim=1)
					topk_vals, topk_indices = combined_vals.topk(min(k, combined_vals.size(1)), dim=1)
					batch_indices = torch.arange(topk_indices.size(0), device=device).unsqueeze(1)
					batch_indices = batch_indices.expand(-1, topk_indices.size(1))
					new_topk_idxs = combined_idxs[batch_indices, topk_indices]
					all_topk_indices[k] = new_topk_idxs
					all_topk_values[k] = topk_vals
			del img_subchunk, similarity, img_labels_subchunk
			torch.cuda.empty_cache()
		
		# Compute micro-average precision@K (equal weight per class)
		for k in topK_values:
			precisions = []
			for c_idx, class_idx in enumerate(chunk_class_indices):
				topk_indices_cpu = all_topk_indices[k][c_idx].cpu()
				retrieved_labels = image_labels[topk_indices_cpu].to(device)
				relevant_count = class_counts[class_idx].item()
				correct = (retrieved_labels == class_idx).sum().item()
				precision = correct / k if relevant_count > 0 else 0.0
				precisions.append(precision)
			# Micro-average: equal weight per class
			avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
			txt2img_topk_accuracy[k] += avg_precision
		del chunk_txt_embeds, all_topk_indices, all_topk_values
		torch.cuda.empty_cache()
		if (i + 1) % print_every == 0 or (i + 1) == num_chunks:
			print(f"  Processed {i+1}/{num_chunks} text chunks")
	txt2img_topk_accuracy = {k: v for k, v in txt2img_topk_accuracy.items()}  # Micro-averaged
	txt2img_acc = txt2img_topk_accuracy[1]  # Align with K=1 precision (top-1 accuracy)
	mean_reciprocal_rank_full = float(np.mean(img2txt_mrr)) if img2txt_mrr else 0.0
	cosine_sim_mean = float(np.mean(cosine_similarities)) if cosine_similarities else 0.0
	
	metrics = {
		"val_loss": float(avg_val_loss),
		"img2txt_acc": float(img2txt_acc),
		"txt2img_acc": float(txt2img_acc),
		"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_accuracy.items()},
		"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_accuracy.items()},
		"mean_reciprocal_rank": mean_reciprocal_rank_full,
		"cosine_similarity": cosine_sim_mean,
	}
	print(f"Elapsed_t: {time.time()-start_time:.1f} sec".center(150, "-"))
	return metrics

def get_validation_metrics_old(
		model: torch.nn.Module,
		validation_loader: torch.utils.data.DataLoader,
		criterion: torch.nn.Module,
		device: str,
		topK_values: List[int],
		cache_dir: str,
		finetune_strategy: str = None,
		chunk_size: int = 1024,
		verbose: bool = True,
		max_in_batch_samples: Optional[int] = None,
		force_recompute: bool = False,
		embeddings_cache: tuple = None,
		lora_params: Optional[Dict] = None,
		is_training: bool = False,
		model_hash: str = None,
	) -> Dict:
	model.eval()
	torch.cuda.empty_cache()
	start_time = time.time()
	
	if finetune_strategy is None:
		finetune_strategy = "pretrained"
	
	model_class_name = model.__class__.__name__
	model_arch_name = getattr(model, 'name', 'unknown_arch')
	try:
		class_names = validation_loader.dataset.dataset.classes
	except AttributeError:
		class_names = validation_loader.dataset.unique_labels
	
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	num_workers = getattr(validation_loader, 'num_workers', 'unknown_num_workers')
	n_classes = len(class_names)
	
	# Construct cache file name with model_hash
	cache_file = os.path.join(
		cache_dir,
		f"{dataset_name}_"
		f"{finetune_strategy}_"
		f"bs_{validation_loader.batch_size}_"
		f"nw_{num_workers}_"
		f"{model_class_name}_"
		f"{re.sub(r'[/@]', '_', model_arch_name)}_"
		f"validation_embeddings.pt"
	)
	if model_hash:
		cache_file = cache_file.replace(".pt", f"_{model_hash}.pt")
	
	# Step 1: Compute in-batch metrics
	in_batch_metrics = None
	if max_in_batch_samples is not None:
		in_batch_start = time.time()
		if verbose:
			print(f"Using {max_in_batch_samples} samples for in-batch metrics computation.")
		in_batch_metrics = compute_direct_in_batch_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topK_values,
			max_samples=max_in_batch_samples
		)
		if verbose:
			print(f"Direct in-batch metrics computed in {time.time() - in_batch_start:.1f} sec")
	else:
		if verbose:
			print("Skipping in-batch metrics computation (max_in_batch_samples is None).")
	
	# Step 2: Load or compute image embeddings
	cache_loaded = False
	all_image_embeds = None
	all_labels = None
	
	if not is_training and embeddings_cache is not None:
			if not isinstance(embeddings_cache, tuple) or len(embeddings_cache) < 2:
					raise ValueError("embeddings_cache must be a tuple of (image_embeds, labels, ...)")
			all_image_embeds, _ = embeddings_cache
			all_labels = torch.tensor([validation_loader.dataset.labels_int[i] for i in range(len(validation_loader.dataset))], device='cpu')
			cache_loaded = True
			if verbose:
					print("Using precomputed embeddings from embeddings_cache")
					print(f"all_image_embeds shape: {all_image_embeds.shape}, dtype: {all_image_embeds.dtype}")
					print(f"all_labels shape: {all_labels.shape}, dtype: {all_labels.dtype}")
	elif not is_training and os.path.exists(cache_file) and not force_recompute:
			if verbose:
					print(f"Loading cached embeddings from {cache_file}")
			try:
					cache_start = time.time()
					cached = torch.load(cache_file, map_location='cpu')
					all_image_embeds = cached.get('image_embeds').to(device)
					all_labels = cached.get('labels').to(device)
					cache_loaded = True
					if verbose:
							print(f"Cache loaded in {time.time() - cache_start:.5f} sec")
			except Exception as e:
					if verbose:
							print(f"Error loading cache: {e}. Computing from scratch.")
					cache_loaded = False
	
	if not cache_loaded or all_image_embeds is None or is_training:
		if verbose:
			print(f"Computing embeddings from scratch Strategy: {finetune_strategy} Model: {model_class_name} Arch: {model_arch_name}...")
		all_image_embeds = []
		all_labels = []
		
		with torch.no_grad():
			for bidx, (images, tokenized_labels, labels_indices) in enumerate(tqdm(validation_loader, desc=f"Encoding images ({finetune_strategy})")):
				images = images.to(device, non_blocking=True) # images is already a batch (Shape: B x C x H x W)
				# ############# using chunk_size to save memory #############
				# for start in range(0, images.size(0), chunk_size):
				# 	end = min(start + chunk_size, images.size(0))
				# 	chunk_images = images[start:end]
				# 	with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				# 		image_embeds = model.encode_image(chunk_images)
				# 	image_embeds = F.normalize(image_embeds, dim=-1).to(torch.float32).cpu()
				# 	all_image_embeds.append(image_embeds)
				# ############# using chunk_size to save memory #############
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					image_embeds = model.encode_image(images)
					image_embeds = F.normalize(image_embeds, dim=-1)
				image_embeds_cpu = image_embeds.to(torch.float32).cpu()
				all_image_embeds.append(image_embeds_cpu)
				all_labels.extend(labels_indices.cpu().tolist())
				del images, image_embeds, image_embeds_cpu
				torch.cuda.empty_cache()
		
		all_image_embeds = torch.cat(all_image_embeds, dim=0)
		all_labels = torch.tensor(all_labels, device='cpu')
		
		if not is_training:
			try:
				cache_content = {'image_embeds': all_image_embeds, 'labels': all_labels}
				torch.save(cache_content, cache_file)
				if verbose: print(f"Saved embeddings to {cache_file}")
			except Exception as e:
				if verbose: print(f"Warning: Failed to save cache: {e}")
	
	# Step 3: Compute class text embeddings
	with torch.no_grad():
		text_inputs = clip.tokenize(class_names).to(device, non_blocking=True)
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			class_text_embeds = model.encode_text(text_inputs)
		class_text_embeds = F.normalize(class_text_embeds, dim=-1).to(torch.float32).cpu()
	
	# Step 4: Compute similarity matrices
	device_image_embeds = all_image_embeds.to(device, dtype=torch.float32)
	device_class_text_embeds = class_text_embeds.to(device, dtype=torch.float32)
	device_labels = all_labels.to(device)
	
	i2t_similarity = device_image_embeds @ device_class_text_embeds.T
	t2i_similarity = device_class_text_embeds @ device_image_embeds.T

	# Step 5: Compute full-set metrics
	full_set_start = time.time()
	full_metrics = compute_full_set_metrics_from_cache(
		i2t_similarity=i2t_similarity,
		t2i_similarity=t2i_similarity,
		labels=device_labels,
		n_classes=n_classes,
		topK_values=topK_values,
		device=device
	)
	if verbose: print(f"Full-set validation metrics computed in {time.time() - full_set_start:.3f} sec")
	
	# Step 6: Compute retrieval metrics
	retrieval_start = time.time()
	cache_key_base = f"{dataset_name}_{finetune_strategy}_{model.__class__.__name__}_{re.sub(r'[/@]', '_', model.name)}"
	if lora_params is not None:
			cache_key_base += f"_lora_rank_{lora_params['lora_rank']}_lora_alpha_{lora_params['lora_alpha']}_lora_dropout_{lora_params['lora_dropout']}"
	
	img2txt_metrics = compute_retrieval_metrics_from_similarity(
			similarity_matrix=i2t_similarity,
			query_labels=device_labels,
			candidate_labels=torch.arange(n_classes, device=device),
			topK_values=topK_values,
			mode="Image-to-Text",
			max_k=n_classes,
			cache_dir=cache_dir,
			cache_key=f"{cache_key_base}_img2txt",
			is_training=is_training,
			verbose=verbose,
	)
	
	class_counts = torch.bincount(device_labels, minlength=n_classes)
	txt2img_metrics = compute_retrieval_metrics_from_similarity(
		similarity_matrix=t2i_similarity,
		query_labels=torch.arange(n_classes, device=device),
		candidate_labels=device_labels,
		topK_values=topK_values,
		mode="Text-to-Image",
		class_counts=class_counts,
		cache_dir=cache_dir,
		cache_key=f"{cache_key_base}_txt2img",
		is_training=is_training,
		verbose=verbose,
	)
	
	if verbose: print(f"Retrieval metrics computed in {time.time() - retrieval_start:.5f} sec")
	
	# Step 7: Return results
	result = {
		"in_batch_metrics": in_batch_metrics,
		"full_metrics": full_metrics,
		"img2txt_metrics": img2txt_metrics,
		"txt2img_metrics": txt2img_metrics
	}
	
	if verbose: print(f"Validation evaluation completed in {time.time() - start_time:.3f} sec")
	
	return result



def plot_image_to_texts_separate_horizontal_bars(
		models: dict,
		validation_loader: DataLoader,
		preprocess,
		img_path: str,
		topk: int,
		device: str,
		results_dir: str,
		figure_size=(15, 6),  # Adjusted for multiple subplots
		dpi: int = 300,  # Increased for publication quality
):
		dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
		print(f"num_models: {len(models)}")
		
		# Prepare labels
		try:
			labels = validation_loader.dataset.dataset.classes
		except AttributeError:
			labels = validation_loader.dataset.unique_labels
		n_labels = len(labels)
		if topk > n_labels:
				print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
				return
		tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
		
		# Load and preprocess image
		try:
				img = Image.open(img_path).convert("RGB")
		except FileNotFoundError:
				try:
						response = requests.get(img_path)
						response.raise_for_status()
						img = Image.open(BytesIO(response.content)).convert("RGB")
				except requests.exceptions.RequestException as e:
						print(f"ERROR: failed to load image from {img_path} => {e}")
						return
		image_tensor = preprocess(img).unsqueeze(0).to(device)

		# Compute predictions for each model
		model_predictions = {}
		model_topk_labels = {}
		model_topk_probs = {}
		for model_name, model in models.items():
				model.eval()
				print(f"[Image-to-text(s)] {model_name} Zero-Shot Image Classification of image: {img_path}".center(200, " "))
				t0 = time.time()
				with torch.no_grad():
						image_features = model.encode_image(image_tensor)
						labels_features = model.encode_text(tokenized_labels_tensor)
						image_features /= image_features.norm(dim=-1, keepdim=True)
						labels_features /= labels_features.norm(dim=-1, keepdim=True)
						similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
				
				# Store full probabilities for all labels
				all_probs = similarities.squeeze().cpu().numpy()
				model_predictions[model_name] = all_probs
				
				# Get top-k labels and probabilities for this model
				topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
				topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
				topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
				topk_pred_labels = [labels[i] for i in topk_pred_indices]
				
				# Sort by descending probability
				sorted_indices = np.argsort(topk_pred_probs)[::-1]
				model_topk_labels[model_name] = [topk_pred_labels[i] for i in sorted_indices]
				model_topk_probs[model_name] = topk_pred_probs[sorted_indices]
				print(f"Top-{topk} predicted labels for {model_name}: {model_topk_labels[model_name]}")
				print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

		# Create subplot grid: 1 row, (1 + len(models)) columns
		num_models = len(models)
		# Adjust figure size dynamically to ensure balanced proportions
		subplot_width = 3  # Approximate width per subplot (in inches)
		fig_width = subplot_width * (1 + num_models)
		fig_height = max(4, topk * 1.2)  # Adjust height based on number of labels
		fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
		gs = gridspec.GridSpec(1, 1 + num_models, width_ratios=[1] * (1 + num_models), wspace=0.1)

		# # Add a suptitle for the entire figure
		# fig.suptitle(
		# 		f"Top-{topk} Predicted Labels for Query Image Across Models",
		# 		fontsize=16, fontweight='bold', y=1.05
		# )

		# Subplot 1: Query Image
		ax0 = plt.subplot(gs[0])
		ax0.imshow(img)
		ax0.axis('off')
		ax0.set_title("Query Image", fontsize=14, fontweight='bold', pad=10)

		# Define colors consistent with plot_comparison_metrics_split/merged
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#7f7f7f'}
		pretrained_model_arch = models.get("pretrained").name
		colors = [pretrained_colors.get(pretrained_model_arch, '#000000')] + list(strategy_colors.values())
		print(f"colors: {colors}")

		# Subplots for each model
		model_names = list(models.keys())
		axes = []
		# Create subplots iteratively to avoid referencing 'axes' before assignment
		for model_idx in range(num_models):
				if model_idx == 0:
						# First subplot shares x-axis with ax0 (image subplot)
						ax = plt.subplot(gs[model_idx + 1])
				else:
						# Subsequent subplots share x-axis with the first model subplot (axes[0])
						ax = plt.subplot(gs[model_idx + 1], sharex=axes[0])
				axes.append(ax)

		for model_idx, (model_name, ax) in enumerate(zip(model_names, axes)):
				y_pos = np.arange(topk)
				sorted_probs = model_topk_probs[model_name]
				sorted_labels = model_topk_labels[model_name]

				# Plot horizontal bars
				bars = ax.barh(
						y_pos,
						sorted_probs,
						color=colors[model_idx],
						edgecolor='white',
						alpha=0.85,
				)
				ax.invert_yaxis()  # Highest probs on top
				# Add y-axis labels (predicted labels)
				ax.set_yticks(y_pos)
				ax.set_yticklabels([label.replace('_', ' ').title() for label in sorted_labels], fontsize=10)
				ax.set_xlim(0, 1)
				# Only set xlabel on the last subplot
				if model_idx == num_models - 1:
						ax.set_xlabel("Probability", fontsize=12, labelpad=10)
				# Add model architecture to pretrained title
				if model_name == "pretrained":
						ax.set_title(f"{model_name.capitalize()} {pretrained_model_arch}", fontsize=14, fontweight='bold', pad=10)
				else:
						ax.set_title(
								model_name.split('_')[-1].replace('finetune', '').capitalize(),
								fontsize=14, fontweight='bold', pad=10
						)
				ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='black')  # Subtler grid
				ax.tick_params(axis='x', labelsize=10)

				# Annotate bars with only the probabilities
				for i, prob in enumerate(sorted_probs):
						ax.text(
								prob + 0.02,
								i,
								f"{prob:.2f}",
								va='center',
								fontsize=10,
								color='black',
								fontweight='bold',
						)

				for spine in ax.spines.values():
						spine.set_color('black')

		plt.tight_layout()

		# Save plot
		img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
		file_name = os.path.join(
				results_dir,
				f'{dataset_name}_Top{topk}_labels_{img_hash}_dataset_separate_bar_image_to_text.png'
		)
		plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
		plt.close()
		print(f"Saved visualization to: {file_name}")

########################### Grok for evaluation
def get_validation_metrics(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    topK_values: List[int],
    finetune_strategy: str,
    cache_dir: str,
    chunk_size: int = 1024,
    verbose: bool = True
) -> Dict:
    """
    Evaluates a CLIP model on a validation set, handling device and dtype consistency.
    """
    model.eval()
    torch.cuda.empty_cache()
    start_time = time.time()

    # Convert string device to torch.device
    if isinstance(device, str):
        device = torch.device(device)

    num_batches = len(validation_loader)
    total_samples = len(validation_loader.dataset)
    try:
        class_names = validation_loader.dataset.dataset.classes
    except:
        class_names = validation_loader.dataset.unique_labels
    n_classes = len(class_names)

    # Log class information
    if verbose:
        print(f"Number of classes: {n_classes}, Class names: {class_names}")

    cache_file = os.path.join(
        cache_dir,
        f"validation_embeddings_{finetune_strategy}_bs_{validation_loader.batch_size}_{model.__class__.__name__}_{re.sub(r'[/@]', '', model.name)}.pt"
    )

    # Initialize metrics
    in_batch_loss = 0.0
    img2txt_correct = 0
    txt2img_correct = 0
    img2txt_topk_correct = {k: 0 for k in topK_values}
    txt2img_topk_correct = {k: 0 for k in topK_values}
    cosine_similarities = []
    full_loss = 0.0

    cache_exists = os.path.exists(cache_file)

    if cache_exists:
        if verbose:
            print(f"Loading cached embeddings from {cache_file}")
        cached = torch.load(cache_file, map_location='cpu')
        all_image_embeds = cached['image_embeds']
        all_text_embeds = cached['text_embeds']
        all_labels = cached['labels']
        class_text_embeds = cached['class_text_embeds']

        # Validate cached labels
        if not (all_labels.min() >= 0 and all_labels.max() < n_classes):
            raise ValueError(
                f"Cached labels contain invalid indices. Expected range [0, {n_classes-1}], "
                f"but found min={all_labels.min()}, max={all_labels.max()}"
            )

        with torch.no_grad():
            # Pre-compute text embeddings for all classes
            text_inputs = clip.tokenize(class_names).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                class_text_embeds = model.encode_text(text_inputs)
            class_text_embeds = F.normalize(class_text_embeds, dim=-1).cpu()
            
            for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
                batch_size = images.size(0)
                # Validate labels_indices for the batch
                if not (labels_indices.min() >= 0 and labels_indices.max() < n_classes):
                    raise ValueError(
                        f"Invalid label indices in batch {bidx+1}: "
                        f"Expected range [0, {n_classes-1}], but found min={labels_indices.min()}, "
                        f"max={labels_indices.max()}"
                    )

                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    chunk_images = images[start:end].to(device, non_blocking=True)
                    chunk_texts = tokenized_labels[start:end].to(device, non_blocking=True)
                    chunk_labels = labels_indices[start:end].to(device, non_blocking=True)
                    chunk_size_actual = end - start

                    # For contrastive loss, use indices within the batch
                    batch_indices = torch.arange(chunk_size_actual, device=device)

                    with torch.amp.autocast(device_type=device.type, enabled=True):
                        # Get image and text embeddings
                        image_embeds = model.encode_image(chunk_images)
                        text_embeds = model.encode_text(chunk_texts)
                        
                        # Normalize embeddings
                        image_embeds = F.normalize(image_embeds, dim=-1)
                        text_embeds = F.normalize(text_embeds, dim=-1)
                        
                        # Compute in-batch contrastive logits
                        logits_per_image = image_embeds @ text_embeds.T
                        logits_per_text = logits_per_image.T
                        
                        # Compute contrastive loss using batch indices
                        loss_img = criterion(logits_per_image, batch_indices)
                        loss_txt = criterion(logits_per_text, batch_indices)

                    # Compute batch loss
                    batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
                    in_batch_loss += batch_loss * (chunk_size_actual / batch_size)
                    full_loss += batch_loss * (chunk_size_actual / total_samples)

                    # Compute in-batch accuracy
                    in_batch_img2txt_pred = logits_per_image.argmax(dim=1)
                    in_batch_txt2img_pred = logits_per_text.argmax(dim=1)
                    img2txt_correct += (in_batch_img2txt_pred == batch_indices).sum().item()
                    txt2img_correct += (in_batch_txt2img_pred == batch_indices).sum().item()

                    # Compute top-k accuracy
                    for k in topK_values:
                        if k <= chunk_size_actual:
                            topk_img2txt = logits_per_image.topk(k, dim=1)[1]
                            topk_txt2img = logits_per_text.topk(k, dim=1)[1]
                            img2txt_topk_correct[k] += sum(batch_indices[i].item() in topk_img2txt[i] for i in range(chunk_size_actual))
                            txt2img_topk_correct[k] += sum(batch_indices[i].item() in topk_txt2img[i] for i in range(chunk_size_actual))

                    # Cosine similarity between paired embeddings
                    cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1).cpu().numpy()
                    cosine_similarities.extend(cos_sim)

                if verbose and ((bidx + 1) % 100 == 0 or bidx + 1 == num_batches):
                    print(f"Processed {bidx + 1}/{num_batches} batches for loss recomputation")
    else:
        # Code path for when cache doesn't exist - collect and save embeddings
        image_embeds_list = []
        text_embeds_list = []
        labels_list = []
        
        with torch.no_grad():
            # Pre-compute text embeddings for all classes
            text_inputs = clip.tokenize(class_names).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                class_text_embeds = model.encode_text(text_inputs)
            class_text_embeds = F.normalize(class_text_embeds, dim=-1).cpu()

            for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
                batch_size = images.size(0)
                # Validate labels_indices
                if not (labels_indices.min() >= 0 and labels_indices.max() < n_classes):
                    raise ValueError(
                        f"Invalid label indices in batch {bidx+1}: "
                        f"Expected range [0, {n_classes-1}], but found min={labels_indices.min()}, "
                        f"max={labels_indices.max()}"
                    )

                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    chunk_images = images[start:end].to(device, non_blocking=True)
                    chunk_texts = tokenized_labels[start:end].to(device, non_blocking=True)
                    chunk_labels = labels_indices[start:end].to(device, non_blocking=True)
                    chunk_size_actual = end - start

                    # For contrastive loss, use batch indices
                    batch_indices = torch.arange(chunk_size_actual, device=device)

                    with torch.amp.autocast(device_type=device.type, enabled=True):
                        # Get image and text embeddings
                        image_embeds = model.encode_image(chunk_images)
                        text_embeds = model.encode_text(chunk_texts)
                        
                        # Normalize embeddings
                        image_embeds = F.normalize(image_embeds, dim=-1)
                        text_embeds = F.normalize(text_embeds, dim=-1)
                        
                        # Compute in-batch contrastive logits
                        logits_per_image = image_embeds @ text_embeds.T
                        logits_per_text = logits_per_image.T
                        
                        # Compute contrastive loss
                        loss_img = criterion(logits_per_image, batch_indices)
                        loss_txt = criterion(logits_per_text, batch_indices)

                    # Compute batch loss
                    batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
                    in_batch_loss += batch_loss * (chunk_size_actual / batch_size)
                    full_loss += batch_loss * (chunk_size_actual / total_samples)

                    # Compute in-batch accuracy
                    in_batch_img2txt_pred = logits_per_image.argmax(dim=1)
                    in_batch_txt2img_pred = logits_per_text.argmax(dim=1)
                    img2txt_correct += (in_batch_img2txt_pred == batch_indices).sum().item()
                    txt2img_correct += (in_batch_txt2img_pred == batch_indices).sum().item()

                    # Compute top-k accuracy
                    for k in topK_values:
                        if k <= chunk_size_actual:
                            topk_img2txt = logits_per_image.topk(k, dim=1)[1]
                            topk_txt2img = logits_per_text.topk(k, dim=1)[1]
                            img2txt_topk_correct[k] += sum(batch_indices[i].item() in topk_img2txt[i] for i in range(chunk_size_actual))
                            txt2img_topk_correct[k] += sum(batch_indices[i].item() in topk_txt2img[i] for i in range(chunk_size_actual))

                    # Store embeddings and labels for full-set evaluation
                    image_embeds_list.append(image_embeds.cpu())
                    text_embeds_list.append(text_embeds.cpu())
                    labels_list.extend(chunk_labels.cpu().tolist())

                    # Cosine similarity between paired embeddings
                    cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1).cpu().numpy()
                    cosine_similarities.extend(cos_sim)

                if verbose and ((bidx + 1) % 100 == 0 or bidx + 1 == num_batches):
                    print(f"Processed {bidx + 1}/{num_batches} batches")

            # Concatenate all embeddings
            all_image_embeds = torch.cat(image_embeds_list, dim=0)
            all_text_embeds = torch.cat(text_embeds_list, dim=0)
            all_labels = torch.tensor(labels_list)

            # Validate all_labels before saving
            if not (all_labels.min() >= 0 and all_labels.max() < n_classes):
                raise ValueError(
                    f"Invalid label indices in collected labels: "
                    f"Expected range [0, {n_classes-1}], but found min={all_labels.min()}, "
                    f"max={all_labels.max()}"
                )

            # Save embeddings to cache
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save({
                'image_embeds': all_image_embeds,
                'text_embeds': all_text_embeds,
                'labels': all_labels,
                'class_text_embeds': class_text_embeds,
            }, cache_file)
            if verbose:
                print(f"Saved embeddings to {cache_file}")

    # In-batch metrics
    in_batch_metrics = {
        "val_loss": in_batch_loss / num_batches,
        "img2txt_acc": img2txt_correct / total_samples,
        "txt2img_acc": txt2img_correct / total_samples,
        "img2txt_topk_acc": {str(k): v / total_samples for k, v in img2txt_topk_correct.items()},
        "txt2img_topk_acc": {str(k): v / total_samples for k, v in txt2img_topk_correct.items()},
        "cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0.0
    }

    # Full-set metrics (class-based) - CRITICAL: Handle device and dtype consistently
    # Move tensors to GPU with consistent type (float32)
    all_image_embeds_gpu = all_image_embeds.to(device, dtype=torch.float32)
    class_text_embeds_gpu = class_text_embeds.to(device, dtype=torch.float32)
    all_labels_gpu = all_labels.to(device)
    
    # Calculate image-to-text similarity
    similarity_i2t = all_image_embeds_gpu @ class_text_embeds_gpu.T
    img2txt_pred = similarity_i2t.argmax(dim=1)
    
    # Ensure tensors are on the same device for comparison
    img2txt_acc = (img2txt_pred == all_labels_gpu).float().mean().item()

    # Calculate image-to-text top-k accuracy
    img2txt_topk_acc = {}
    valid_k_values = [k for k in topK_values if k <= n_classes]
    for k in valid_k_values:
        topk_indices = similarity_i2t.topk(k, dim=1)[1]
        correct = (topk_indices == all_labels_gpu.unsqueeze(1)).any(dim=1)
        img2txt_topk_acc[k] = correct.float().mean().item()

    # Calculate text-to-image similarity
    similarity_t2i = class_text_embeds_gpu @ all_image_embeds_gpu.T
    txt2img_topk_acc = {}
    
    # Create class index tensor on the same device
    class_indices = torch.arange(n_classes, device=device)
    
    # Calculate text-to-image accuracy
    txt2img_acc = similarity_t2i.argmax(dim=1).eq(class_indices).float().mean().item()
    
    # Calculate text-to-image top-k accuracy
    for k in topK_values:
        if k > total_samples:
            txt2img_topk_acc[k] = 0.0  # Skip if k exceeds available samples
            continue
            
        topk_indices = similarity_t2i.topk(k, dim=1)[1]
        correct = 0
        for i in range(n_classes):
            # Get labels of top-k retrieved images
            retrieved_labels = all_labels_gpu[topk_indices[i]]
            # Check if class i is in retrieved labels
            if (retrieved_labels == i).any().item():
                correct += 1
        txt2img_topk_acc[k] = correct / n_classes

    # Calculate mean reciprocal rank
    ranks = similarity_i2t.argsort(dim=1, descending=True)
    rr_indices = ranks.eq(all_labels_gpu.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
    img2txt_mrr = (1.0 / rr_indices.float()).mean().item()

    # Construct full metrics dictionary
    full_metrics = {
        "val_loss": full_loss,
        "img2txt_acc": img2txt_acc,
        "txt2img_acc": txt2img_acc,
        "img2txt_topk_acc": {str(k): v for k, v in img2txt_topk_acc.items()},
        "txt2img_topk_acc": {str(k): v for k, v in txt2img_topk_acc.items()},
        "mean_reciprocal_rank": img2txt_mrr,
        "cosine_similarity": in_batch_metrics["cosine_similarity"]
    }

    # Retrieval metrics
    class_counts = torch.bincount(all_labels, minlength=n_classes).to(device)
    
    # All tensors should now be on the device with consistent dtype
    img2txt_metrics = get_retrieval_metrics(
        similarity_matrix=similarity_i2t,
        query_labels=all_labels_gpu,
        candidate_labels=class_indices,
        topK_values=topK_values,
        mode="Image-to-Text",
        class_counts=None,
        max_k=n_classes
    )
    
    txt2img_metrics = get_retrieval_metrics(
        similarity_matrix=similarity_t2i,
        query_labels=class_indices,
        candidate_labels=all_labels_gpu,
        topK_values=topK_values,
        mode="Text-to-Image",
        class_counts=class_counts,
        max_k=None
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"Evaluation completed in {elapsed:.2f} seconds")

    # Free GPU memory
    del all_image_embeds_gpu, class_text_embeds_gpu, all_labels_gpu
    del similarity_i2t, similarity_t2i
    torch.cuda.empty_cache()

    return {
        "in_batch_metrics": in_batch_metrics,
        "full_metrics": full_metrics,
        "img2txt_metrics": img2txt_metrics,
        "txt2img_metrics": txt2img_metrics
    }

def compute_in_batch_metrics(
		validation_loader: DataLoader,
		image_embeds: torch.Tensor,
		text_embeds: torch.Tensor,
		labels: torch.Tensor,
		total_loss: float,
		num_batches: int,
		total_samples: int,
		topK_values: List[int],
		n_classes: int,
		device: str,
		cosine_similarities: List[float]
) -> Dict:
		"""
		Compute in-batch metrics including loss, accuracy, and top-K accuracy using updated embeddings.
		"""
		avg_loss = total_loss / num_batches
		valid_k_values = [k for k in topK_values if k <= n_classes]

		img2txt_topk_acc = {k: 0 for k in valid_k_values}
		txt2img_topk_acc = {k: 0 for k in topK_values}
		img2txt_correct = 0
		txt2img_correct = 0

		# Process embeddings in chunks to simulate batch-wise computation
		batch_size = validation_loader.batch_size
		for start in range(0, total_samples, batch_size):
				end = min(start + batch_size, total_samples)
				chunk_img_embeds = image_embeds[start:end].to(device)
				chunk_txt_embeds = text_embeds[start:end].to(device)
				chunk_labels = labels[start:end].to(device)
				chunk_size = end - start

				similarity = chunk_img_embeds @ chunk_txt_embeds.T
				img2txt_pred = similarity.argmax(dim=1)
				txt2img_pred = similarity.T.argmax(dim=1)

				for i in range(chunk_size):
						pred_label = chunk_labels[img2txt_pred[i]].item()
						true_label = chunk_labels[i].item()
						if pred_label == true_label:
								img2txt_correct += 1
						pred_label = chunk_labels[txt2img_pred[i]].item()
						if pred_label == true_label:
								txt2img_correct += 1

				for k in valid_k_values:
						topk_img2txt = similarity.topk(min(k, chunk_size), dim=1)[1]
						for i in range(chunk_size):
								topk_labels = chunk_labels[topk_img2txt[i]].cpu()
								if chunk_labels[i].item() in topk_labels:
										img2txt_topk_acc[k] += 1
						topk_txt2img = similarity.T.topk(min(k, chunk_size), dim=1)[1]
						for i in range(chunk_size):
								topk_labels = chunk_labels[topk_txt2img[i]].cpu()
								if chunk_labels[i].item() in topk_labels:
										txt2img_topk_acc[k] += 1

		img2txt_acc = img2txt_correct / total_samples
		txt2img_acc = txt2img_correct / total_samples
		img2txt_topk_acc = {k: v / total_samples for k, v in img2txt_topk_acc.items()}
		txt2img_topk_acc = {k: v / total_samples for k, v in txt2img_topk_acc.items()}

		return {
				"val_loss": float(avg_loss),
				"img2txt_acc": float(img2txt_acc),
				"txt2img_acc": float(txt2img_acc),
				"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
				"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
				"cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0.0
		}

def compute_full_metrics(
		image_embeds: torch.Tensor,
		text_embeds: torch.Tensor,
		labels: torch.Tensor,
		class_text_embeds: torch.Tensor,
		total_loss: float,
		num_batches: int,
		total_samples: int,
		topK_values: List[int],
		n_classes: int,
		device: str,
		cosine_similarities: List[float]
) -> Dict:
		"""
		Compute full-set metrics including loss, accuracy, top-K accuracy, and MRR using updated embeddings.
		"""
		avg_loss = total_loss / num_batches
		valid_k_values = [k for k in topK_values if k <= n_classes]

		# Image-to-text metrics
		similarity = image_embeds.to(device) @ class_text_embeds.to(device).T
		img2txt_pred = similarity.argmax(dim=1)
		img2txt_acc = (img2txt_pred.cpu() == labels).float().mean().item()

		img2txt_topk_acc = {}
		img2txt_mrr = []
		for k in valid_k_values:
				topk_indices = similarity.topk(k, dim=1)[1]
				correct = (topk_indices == labels.unsqueeze(1).to(device)).any(dim=1)
				img2txt_topk_acc[k] = correct.float().mean().item()

				ranks = similarity.argsort(dim=1, descending=True)
				rr_indices = ranks.eq(labels.view(-1, 1).to(device)).nonzero(as_tuple=True)[1] + 1
				img2txt_mrr.extend((1.0 / rr_indices.float()).cpu().numpy())

		# Text-to-image metrics
		similarity_t2i = class_text_embeds.to(device) @ image_embeds.to(device).T
		txt2img_topk_acc = {}
		for k in topK_values:
				topk_indices = similarity_t2i.topk(k, dim=1)[1].cpu()
				correct = 0
				for i in range(n_classes):
						retrieved_labels = labels[topk_indices[i]]
						true_label = i
						if true_label in retrieved_labels:
								correct += 1
				txt2img_topk_acc[k] = correct / n_classes  # Micro-average precision per class
		txt2img_acc = txt2img_topk_acc.get(1, 0.0)

		return {
				"val_loss": float(avg_loss),
				"img2txt_acc": float(img2txt_acc),
				"txt2img_acc": float(txt2img_acc),
				"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
				"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
				"mean_reciprocal_rank": float(np.mean(img2txt_mrr)) if img2txt_mrr else 0.0,
				"cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0.0
		}

def compute_retrieval_metrics(
		image_embeds: torch.Tensor,
		text_embeds: torch.Tensor,
		labels: torch.Tensor,
		class_text_embeds: torch.Tensor,
		topK_values: List[int],
		n_classes: int,
		device: str
) -> Dict:
		"""
		Compute retrieval metrics (mP@K, mAP@K, Recall@K) using updated embeddings.
		"""
		similarity_i2t = image_embeds.to(device) @ class_text_embeds.to(device).T
		class_counts = torch.bincount(labels, minlength=n_classes)

		img2txt_metrics = get_retrieval_metrics(
				similarity_matrix=similarity_i2t,
				query_labels=labels,
				candidate_labels=torch.arange(n_classes),
				topK_values=topK_values,
				mode="Image-to-Text",
				class_counts=None,
				max_k=n_classes
		)

		similarity_t2i = class_text_embeds.to(device) @ image_embeds.to(device).T
		txt2img_metrics = get_retrieval_metrics(
				similarity_matrix=similarity_t2i,
				query_labels=torch.arange(n_classes),
				candidate_labels=labels,
				topK_values=topK_values,
				mode="Text-to-Image",
				class_counts=class_counts,
				max_k=None
		)

		return {"img2txt": img2txt_metrics, "txt2img": txt2img_metrics}

############################################################################################################################


#### Claude
def evaluate_validation_set(
        model: torch.nn.Module,
        validation_loader: DataLoader,
        criterion: torch.nn.Module,
        device: str,
        topK_values: List[int],
        finetune_strategy: str,
        cache_dir: str,
        chunk_size: int = 1024,
        verbose: bool = True,
        max_in_batch_samples: int = 320  # Limit in-batch computation for speed
) -> Dict:
    """
    Unified validation function to compute in-batch, full-set, and retrieval metrics with proper caching.
    
    This function optimizes validation by:
    1. Using cached embeddings as a starting point
    2. Recomputing needed values for the current model state
    3. Ensuring accuracy by directly computing metrics on a subset for in-batch calculations
    """
    model.eval()
    torch.cuda.empty_cache()
    start_time = time.time()

    # Initialize storage
    all_image_embeds = []
    all_text_embeds = []
    all_labels = []
    in_batch_loss = 0.0
    full_loss = 0.0
    num_batches = len(validation_loader)
    total_samples = len(validation_loader.dataset)
    cosine_similarities = []

    try:
        class_names = validation_loader.dataset.dataset.classes
    except:
        class_names = validation_loader.dataset.unique_labels
    n_classes = len(class_names)

    # Define cache file with unique naming
    cache_file = os.path.join(
            cache_dir,
            f"validation_embeddings_"
            f"{finetune_strategy}_"
            f"bs_{validation_loader.batch_size}_"
            f"{model.__class__.__name__}_"
            f"{re.sub(r'[/@]', '', model.name)}.pt"
    )

    # --- Step 1: Compute small-batch in-batch metrics directly for accuracy ---
    # This gives us accurate in-batch metrics using the current model
    in_batch_metrics = compute_direct_in_batch_metrics(
        model=model,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device,
        topK_values=topK_values,
        max_samples=max_in_batch_samples
    )
    if verbose:
        print(f"Direct in-batch metrics computed in {time.time() - start_time:.1f} sec")

    # --- Step 2: Handle cached embeddings for full-set metrics ---
    cache_start_time = time.time()
    cache_loaded = False
    
    # Load cached embeddings as a starting point if available
    if cache_file and os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached embeddings from {cache_file}")
        try:
            cached = torch.load(cache_file, map_location='cpu')
            all_image_embeds = cached['image_embeds']
            all_text_embeds = cached['text_embeds']
            all_labels = cached['labels']
            class_text_embeds = cached['class_text_embeds']
            cache_loaded = True
            
            if verbose:
                print(f"Cache loaded in {time.time() - cache_start_time:.1f} sec")
        except Exception as e:
            if verbose:
                print(f"Error loading cache: {e}. Will compute from scratch.")
            cache_loaded = False

    # --- Step 3: Process embeddings (either from cache or from scratch) ---
    embed_start_time = time.time()
    
    if cache_loaded:
        # Re-encode all embeddings with current model state
        with torch.no_grad():
            # Recompute class text embeddings with current model
            text_inputs = clip.tokenize(class_names).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                class_text_embeds = model.encode_text(text_inputs)
            class_text_embeds = F.normalize(class_text_embeds, dim=-1).cpu()
            
            # Process validation set in chunks for efficiency
            processed_chunks = 0
            for start_idx in range(0, len(all_labels), chunk_size):
                end_idx = min(start_idx + chunk_size, len(all_labels))
                chunk_size_actual = end_idx - start_idx
                processed_chunks += 1
                
                if verbose and processed_chunks % 5 == 0:
                    print(f"Processing chunk {processed_chunks}... " 
                          f"({start_idx+1}-{end_idx}/{len(all_labels)})")
                
                # We only need to compute full-set similarity
                # No need to recompute all_image_embeds and all_text_embeds
                # Just use them to compute full-set metrics
    else:
        # Compute embeddings and loss from scratch
        with torch.no_grad():
            # Compute class text embeddings
            text_inputs = clip.tokenize(class_names).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                class_text_embeds = model.encode_text(text_inputs)
            class_text_embeds = F.normalize(class_text_embeds, dim=-1).cpu()

            # Single pass through validation set
            for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
                batch_size = images.size(0)
                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    chunk_images = images[start:end].to(device, non_blocking=True)
                    chunk_texts = tokenized_labels[start:end].to(device, non_blocking=True)
                    chunk_labels = labels_indices[start:end].to(device, non_blocking=True)
                    chunk_size_actual = end - start

                    with torch.amp.autocast(device_type=device.type, enabled=True):
                        logits_per_image, logits_per_text = model(chunk_images, chunk_texts)
                        image_embeds = model.encode_image(chunk_images)
                        text_embeds = model.encode_text(chunk_texts)

                    # In-batch loss calculation (not used for metrics, but included for cache)
                    ground_truth = torch.arange(chunk_size_actual, device=device)
                    loss_img = criterion(logits_per_image, ground_truth)
                    loss_txt = criterion(logits_per_text, ground_truth)
                    batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
                    full_loss += batch_loss * (chunk_size_actual / batch_size)

                    # Normalize and store embeddings
                    image_embeds = F.normalize(image_embeds, dim=-1).cpu()
                    text_embeds = F.normalize(text_embeds, dim=-1).cpu()
                    all_image_embeds.append(image_embeds)
                    all_text_embeds.append(text_embeds)
                    all_labels.extend(chunk_labels.cpu().tolist())

                # Print progress
                if verbose and ((bidx + 1) % 10 == 0 or bidx + 1 == num_batches):
                    print(f"Processed {bidx + 1}/{num_batches} batches")

            # Concatenate embeddings
            all_image_embeds = torch.cat(all_image_embeds, dim=0)
            all_text_embeds = torch.cat(all_text_embeds, dim=0)
            all_labels = torch.tensor(all_labels)

            # Save embeddings to cache
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_content = {
                'image_embeds': all_image_embeds,
                'text_embeds': all_text_embeds,
                'labels': all_labels,
                'class_text_embeds': class_text_embeds,
            }
            torch.save(cache_content, cache_file)
            if verbose:
                print(f"Saved embeddings to {cache_file}")

    if verbose:
        print(f"Embeddings processed in {time.time() - embed_start_time:.1f} sec")

    # --- Step 4: Compute full-set metrics with current model state ---
    metrics_start_time = time.time()
    
    # Convert to device for computation
    device_image_embeds = all_image_embeds.to(device)
    device_class_text_embeds = class_text_embeds.to(device)
    device_labels = all_labels.to(device) if isinstance(all_labels, torch.Tensor) else torch.tensor(all_labels, device=device)
    
    # Compute similarity matrix using current embeddings
    similarity_i2t = device_image_embeds @ device_class_text_embeds.T
    
    # Image-to-Text accuracy (full-set)
    img2txt_preds = similarity_i2t.argmax(dim=1)
    img2txt_acc = (img2txt_preds == device_labels).float().mean().item()
    
    # Image-to-Text top-K accuracy (full-set)
    img2txt_topk_acc = {}
    valid_k_values = [k for k in topK_values if k <= n_classes]
    
    for k in valid_k_values:
        topk_indices = similarity_i2t.topk(k, dim=1)[1]
        correct = (topk_indices == device_labels.unsqueeze(1)).any(dim=1)
        img2txt_topk_acc[k] = correct.float().mean().item()
    
    # Text-to-Image metrics (using class embeddings)
    similarity_t2i = device_class_text_embeds @ device_image_embeds.T
    txt2img_topk_acc = {}
    
    for k in topK_values:
        class_correct = 0
        effective_k = min(k, len(device_image_embeds))
        
        topk_indices = similarity_t2i.topk(effective_k, dim=1)[1].cpu()
        for class_idx in range(n_classes):
            retrieved_labels = all_labels[topk_indices[class_idx]]
            if class_idx in retrieved_labels:
                class_correct += 1
        
        txt2img_topk_acc[k] = class_correct / n_classes
    
    txt2img_acc = txt2img_topk_acc.get(1, 0.0)
    
    # Image-to-Text MRR
    ranks = similarity_i2t.argsort(dim=1, descending=True)
    rr_indices = ranks.eq(device_labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
    img2txt_mrr = (1.0 / rr_indices.float()).mean().item()
    
    # Package full-set metrics
    full_metrics = {
        "val_loss": float(in_batch_metrics["val_loss"]),  # Use the more accurate loss from direct computation
        "img2txt_acc": float(img2txt_acc),
        "txt2img_acc": float(txt2img_acc),
        "img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
        "txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
        "mean_reciprocal_rank": float(img2txt_mrr),
        "cosine_similarity": float(in_batch_metrics["cosine_similarity"])  # Use direct computation result
    }
    
    if verbose:
        print(f"Full-set metrics computed in {time.time() - metrics_start_time:.1f} sec")

    # --- Step 5: Compute retrieval metrics ---
    retrieval_start_time = time.time()
    
    # Reuse the similarity matrices we just computed
    img2txt_metrics = compute_retrieval_metrics_from_similarity(
        similarity_matrix=similarity_i2t,
        query_labels=device_labels,
        candidate_labels=torch.arange(n_classes, device=device),
        topK_values=topK_values,
        mode="Image-to-Text",
        max_k=n_classes
    )
    
    # Get class counts for text-to-image retrieval
    class_counts = torch.bincount(device_labels, minlength=n_classes)
    
    txt2img_metrics = compute_retrieval_metrics_from_similarity(
        similarity_matrix=similarity_t2i,
        query_labels=torch.arange(n_classes, device=device),
        candidate_labels=device_labels,
        topK_values=topK_values,
        mode="Text-to-Image",
        class_counts=class_counts
    )
    
    if verbose:
        print(f"Retrieval metrics computed in {time.time() - retrieval_start_time:.1f} sec")

    # --- Step 6: Report and return results ---
    if verbose:
        print("\n--- Validation Metrics ---")
        print("In-batch Metrics:")
        print(json.dumps(in_batch_metrics, indent=2, ensure_ascii=False))
        print("Full-set Metrics:")
        print(json.dumps(full_metrics, indent=2, ensure_ascii=False))
        print("Image-to-Text Retrieval:")
        print(json.dumps(img2txt_metrics, indent=2, ensure_ascii=False))
        print("Text-to-Image Retrieval:")
        print(json.dumps(txt2img_metrics, indent=2, ensure_ascii=False))
        print(f"Validation evaluation completed in {time.time() - start_time:.1f} sec")

    return {
        "in_batch_metrics": in_batch_metrics,
        "full_metrics": full_metrics,
        "img2txt_metrics": img2txt_metrics,
        "txt2img_metrics": txt2img_metrics
    }

def compute_direct_in_batch_metrics(
        model: torch.nn.Module,
        validation_loader: DataLoader,
        criterion: torch.nn.Module,
        device: str,
        topK_values: List[int],
        max_samples: int = 320  # Limit to this many samples for speed
) -> Dict:
    """
    Compute in-batch metrics directly using the current model state.
    This is more accurate than using cached embeddings but limited to a subset
    of validation data for efficiency.
    """
    model.eval()
    total_loss = 0.0
    total_img2txt_correct = 0
    total_txt2img_correct = 0
    processed_batches = 0
    total_samples = 0
    cosine_similarities = []
    
    try:
        class_names = validation_loader.dataset.dataset.classes
    except:
        class_names = validation_loader.dataset.unique_labels
    
    n_classes = len(class_names)
    valid_k_values = [k for k in topK_values if k <= n_classes]
    
    img2txt_topk_accuracy = {k: 0 for k in valid_k_values}
    txt2img_topk_accuracy = {k: 0 for k in topK_values}
    
    with torch.no_grad():
        for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
            # Stop processing if we've reached max_samples
            if total_samples >= max_samples:
                break
                
            batch_size = images.size(0)
            # Adjust batch size if adding the whole batch would exceed max_samples
            if total_samples + batch_size > max_samples:
                effective_batch_size = max_samples - total_samples
                images = images[:effective_batch_size]
                tokenized_labels = tokenized_labels[:effective_batch_size]
                labels_indices = labels_indices[:effective_batch_size]
                batch_size = effective_batch_size
            
            images = images.to(device, non_blocking=True)
            tokenized_labels = tokenized_labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits_per_image, logits_per_text = model(images, tokenized_labels)
                
                # Ground truth for contrastive loss: diagonal of the similarity matrix
                ground_truth = torch.arange(batch_size, device=device)
                
                # Compute loss
                loss_img = criterion(logits_per_image, ground_truth)
                loss_txt = criterion(logits_per_text, ground_truth)
                batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
                total_loss += batch_loss
            
            # Top-1 image-to-text accuracy
            img2txt_preds = torch.argmax(logits_per_image, dim=1)
            img2txt_correct = (img2txt_preds == ground_truth).sum().item()
            total_img2txt_correct += img2txt_correct
            
            # Top-1 text-to-image accuracy
            txt2img_preds = torch.argmax(logits_per_text, dim=1)
            txt2img_correct = (txt2img_preds == ground_truth).sum().item()
            total_txt2img_correct += txt2img_correct
            
            # Top-K accuracy for image-to-text
            for k in valid_k_values:
                topk_preds = torch.topk(logits_per_image, k=min(k, batch_size), dim=1)[1]
                topk_correct = sum([(ground_truth[i] in topk_preds[i]) for i in range(batch_size)])
                img2txt_topk_accuracy[k] += topk_correct
            
            # Top-K accuracy for text-to-image
            for k in topK_values:
                topk_preds = torch.topk(logits_per_text, k=min(k, batch_size), dim=1)[1]
                topk_correct = sum([(ground_truth[i] in topk_preds[i]) for i in range(batch_size)])
                txt2img_topk_accuracy[k] += topk_correct
            
            # Compute cosine similarity between corresponding image and text embeddings
            with torch.amp.autocast(device_type=device.type, enabled=True):
                image_embeds = model.encode_image(images)
                text_embeds = model.encode_text(tokenized_labels)
            
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            # Calculate cosine similarity for each image-text pair
            for i in range(batch_size):
                cos_sim = F.cosine_similarity(image_embeds[i:i+1], text_embeds[i:i+1], dim=-1).item()
                cosine_similarities.append(cos_sim)
            
            processed_batches += 1
            total_samples += batch_size
    
    # Calculate final metrics
    if total_samples == 0:
        return {
            "val_loss": 0.0,
            "img2txt_acc": 0.0,
            "txt2img_acc": 0.0,
            "img2txt_topk_acc": {str(k): 0.0 for k in valid_k_values},
            "txt2img_topk_acc": {str(k): 0.0 for k in topK_values},
            "cosine_similarity": 0.0
        }
    
    avg_loss = total_loss / processed_batches
    img2txt_acc = total_img2txt_correct / total_samples
    txt2img_acc = total_txt2img_correct / total_samples
    
    img2txt_topk_acc = {k: v / total_samples for k, v in img2txt_topk_accuracy.items()}
    txt2img_topk_acc = {k: v / total_samples for k, v in txt2img_topk_accuracy.items()}
    
    avg_cos_sim = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
    
    return {
        "val_loss": float(avg_loss),
        "img2txt_acc": float(img2txt_acc),
        "txt2img_acc": float(txt2img_acc),
        "img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
        "txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
        "cosine_similarity": float(avg_cos_sim)
    }

def compute_retrieval_metrics_from_similarity(
        similarity_matrix: torch.Tensor,
        query_labels: torch.Tensor,
        candidate_labels: torch.Tensor,
        topK_values: List[int],
        mode: str = "Image-to-Text",
        class_counts: torch.Tensor = None,
        max_k: int = None
) -> Dict:
    """
    Compute retrieval metrics (mP@K, mAP@K, Recall@K) using pre-computed similarity matrix.
    This is a streamlined version of get_retrieval_metrics that focuses on performance.
    """
    num_queries, num_candidates = similarity_matrix.shape
    
    # Filter K values if max_k is specified
    if max_k is not None:
        valid_K_values = [K for K in topK_values if K <= max_k]
    else:
        valid_K_values = topK_values
    
    metrics = {
        "mP": {},
        "mAP": {},
        "Recall": {},
    }
    
    # Sort once for all K values to improve efficiency
    all_sorted_indices = torch.argsort(-similarity_matrix, dim=1)
    
    for K in valid_K_values:
        top_k_indices = all_sorted_indices[:, :K]
        
        precision, recall, ap = [], [], []
        for i in range(num_queries):
            true_label = query_labels[i]
            retrieved_labels = candidate_labels[top_k_indices[i]]
            
            # Count correct items in top-K 
            correct_mask = (retrieved_labels == true_label)
            correct = correct_mask.sum().item()
            
            # 1. Precision @ K
            precision.append(correct / K)
            
            # 2. Compute Recall@K
            if mode == "Image-to-Text":
                relevant_count = 1  # Single relevant item per query
            else:
                relevant_count = (
                    class_counts[true_label].item()
                    if class_counts is not None
                    else 0
                )
            
            if relevant_count > 0:
                recall.append(correct / relevant_count)
            else:
                recall.append(0.0)
            
            # 3. Compute AP@K
            relevant_positions = torch.where(correct_mask)[0]
            p_at = []
            cumulative_correct = 0
            
            for pos in relevant_positions:
                if pos < K:  # Only consider positions within top-K
                    cumulative_correct += 1
                    precision_at_rank = cumulative_correct / (pos + 1)
                    p_at.append(precision_at_rank)
            
            # Determine normalization factor for AP
            if mode == "Image-to-Text":
                R = 1  # Always 1 relevant item for image-to-text
            else:
                R = (
                    class_counts[true_label].item()
                    if class_counts is not None
                    else 0
                )
            
            # Handle edge cases
            if R == 0 or len(p_at) == 0:
                ap.append(0.0)
            else:
                ap.append(sum(p_at) / min(R, K))
        
        # Store metrics for this K
        metrics["mP"][str(K)] = torch.tensor(precision).mean().item()
        metrics["mAP"][str(K)] = torch.tensor(ap).mean().item()
        metrics["Recall"][str(K)] = torch.tensor(recall).mean().item()
    
    return metrics

def plot_comparison_metrics_split(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,
		finetune_strategy: str,
		results_dir: str,
		topK_values: list,
		figure_size=(7, 7),
		DPI: int = 300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = [
		'RN50', 
		'RN101', 
		'RN50x4', 
		'RN50x16', 
		'RN50x64', 
		'ViT-B/32', 
		'ViT-B/16', 
		'ViT-L/14', 
		'ViT-L/14@336px',
	]

	if model_name not in finetuned_img2txt_dict.keys():
		print(f"WARNING: {model_name} not found in finetuned_img2txt_dict. Skipping...")
		print(json.dumps(finetuned_img2txt_dict, indent=4, ensure_ascii=False))
		return
	if model_name not in finetuned_txt2img_dict.keys():
		print(f"WARNING: {model_name} not found in finetuned_txt2img_dict. Skipping...")
		print(json.dumps(finetuned_txt2img_dict, indent=4, ensure_ascii=False))
		return
	model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
	model_colors = plt.cm.tab10.colors

	for mode in modes:
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		print(f"\n{'='*80}")
		print(f"ANALYSIS: {mode} | Model: {model_name} | Finetune: {finetune_strategy}")
		print(f"{'='*80}")
		for metric in metrics:
			print(f"\nMetric: {metric}")
			fig, ax = plt.subplots(figsize=figure_size)
			fname = f"{dataset_name}_{finetune_strategy}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_{mode.replace('-', '_')}_{metric}_comparison.png"
			file_path = os.path.join(results_dir, fname)
			
			# Fix: Check if metric exists in both dictionaries with proper nesting
			if metric not in pretrained_dict.get(model_name, {}):
				print(f"WARNING: Metric {metric} not found in pretrained_{mode.lower().replace('-', '_')}_dict for {model_name}")
				continue
				
			if finetune_strategy not in finetuned_dict.get(model_name, {}) or metric not in finetuned_dict.get(model_name, {}).get(finetune_strategy, {}):
				print(f"WARNING: Metric {metric} not found in finetuned_{mode.lower().replace('-', '_')}_dict for {model_name}/{finetune_strategy}")
				continue
			
			# Get available k values that exist in both dictionaries
			k_values = sorted(
				k for k in topK_values if
				str(k) in pretrained_dict.get(model_name, {}).get(metric, {}) and
				str(k) in finetuned_dict.get(model_name, {}).get(finetune_strategy, {}).get(metric, {})
			)
			
			if not k_values:
				print(f"WARNING: No matching K values found for {metric}")
				continue
				
			pretrained_vals = [pretrained_dict[model_name][metric].get(str(k), float('nan')) for k in k_values]
			finetuned_vals = [finetuned_dict[model_name][finetune_strategy][metric].get(str(k), float('nan')) for k in k_values]
			
			# Plot Pre-trained
			ax.plot(
				k_values,
				pretrained_vals,
				label=f"Pre-trained CLIP {model_name}",
				color=model_colors[model_name_idx],
				linestyle='--',
				marker='o',
				linewidth=2,
				alpha=0.7
			)
			# Plot Fine-tuned
			ax.plot(
				k_values,
				finetuned_vals,
				label=f"{finetune_strategy.capitalize()} Fine-tune",
				color=model_colors[model_name_idx],
				linestyle='-',
				marker='s',
				linewidth=2
			)
			# Add annotation for key Ks
			for k in [1, 10, 20]:
				if k in k_values:
					pre = pretrained_dict[model_name][metric][str(k)]
					fine = finetuned_dict[model_name][finetune_strategy][metric][str(k)]
					imp = (fine - pre) / pre * 100 if pre != 0 else 0
					color = 'darkgreen' if imp >= 0 else 'red'
					ax.annotate(
						f"{imp:+.1f}%",
						xy=(k, fine),
						xytext=(5, 5),
						textcoords='offset points',
						fontsize=9,
						fontweight='bold',
						color=color,
						bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3)
					)
			# Axes formatting
			ax.set_title(f"{mode}: {metric}@K", fontsize=12, fontweight='bold')
			ax.set_xlabel("K", fontsize=11)
			ax.set_xticks(k_values)
			ax.grid(True, linestyle='--', alpha=0.9)
			ax.set_ylim(bottom=-0.01, top=1.01)
			ax.legend(fontsize=9, loc='best')
			# Save and close
			plt.tight_layout()
			plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
			plt.close(fig)
			print(f"Saved: {file_path}")

def plot_comparison_metrics_merged(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,  # e.g., 'ViT-B/32'
		finetune_strategy: str,  # e.g., 'full'
		results_dir: str,
		topK_values: list,
		figure_size=(14, 5),
		DPI: int = 300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	# Validate model_name and finetune_strategy
	if model_name not in finetuned_img2txt_dict or finetune_strategy not in finetuned_img2txt_dict.get(model_name, {}):
			print(f"WARNING: {finetune_strategy} for {model_name} not found in finetuned_img2txt_dict. Skipping...")
			return
	if model_name not in finetuned_txt2img_dict or finetune_strategy not in finetuned_txt2img_dict.get(model_name, {}):
			print(f"WARNING: {finetune_strategy} for {model_name} not found in finetuned_txt2img_dict. Skipping...")
			return
	model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
	model_colors = plt.cm.tab10.colors
	# Print detailed analysis information to logs
	print(f"\n{'='*80}")
	print(f"DETAILED PERFORMANCE ANALYSIS FOR {dataset_name} - {model_name} WITH {finetune_strategy.capitalize()} FINE-TUNING")
	print(f"{'='*80}")
	# Create separate figures for each mode
	for i, mode in enumerate(modes):
			fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
			fname = f"{dataset_name}_{finetune_strategy}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_retrieval_performance_comparison_{mode.replace('-', '_')}_merged.png"
			file_path = os.path.join(results_dir, fname)
			fig.suptitle(
					f'$\\it{{{mode}}}$ Retrieval Performance Comparison\n'
					f'Pre-trained CLIP {model_name} vs. {finetune_strategy.capitalize()} Fine-tuning',
					fontsize=12,
					fontweight='bold',
			)
			# Select the appropriate dictionaries
			pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
			finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
			print(f"\n{'-'*40}")
			print(f"MODE: {mode}: {fname}")
			print(f"{'-'*40}")
			for j, metric in enumerate(metrics):
					ax = axes[j]
					print(f"\nMetric: {metric}")
					print("-" * 20)
					# Compute aligned k_values across both dictionaries
					k_values = sorted(
							k for k in topK_values if
							str(k) in pretrained_dict.get(model_name, {}).get(metric, {}) and
							str(k) in finetuned_dict.get(model_name, {}).get(finetune_strategy, {}).get(metric, {})
					)
					if not k_values:
							print(f"WARNING: No matching K values for {mode}, {metric}")
							continue
					# Extract values safely with .get()
					pretrained_values = [pretrained_dict.get(model_name, {}).get(metric, {}).get(str(k), float('nan')) for k in k_values]
					finetuned_values = [finetuned_dict.get(model_name, {}).get(finetune_strategy, {}).get(metric, {}).get(str(k), float('nan')) for k in k_values]
					# Plot pre-trained model performance
					ax.plot(
							k_values,
							pretrained_values,
							label=f"Pre-trained CLIP {model_name}",
							color=model_colors[model_name_idx],
							marker='o',
							linestyle='--',
							linewidth=2,
							markersize=5,
							alpha=0.7,
					)
					# Plot fine-tuned model performance
					ax.plot(
							k_values,
							finetuned_values,
							label=f"{finetune_strategy.capitalize()} Fine-tune",
							color=model_colors[model_name_idx],
							marker='s',
							linestyle='-',
							linewidth=2,
							markersize=5,
					)
					# Add improvement percentages at key points
					key_k_values = [1, 10, 20]  # Annotate these K values if available
					for k in key_k_values:
							if k in k_values:
									k_idx = k_values.index(k)
									pretrained_val = pretrained_values[k_idx]
									finetuned_val = finetuned_values[k_idx]
									if pretrained_val != 0:  # Avoid division by zero
											improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
											text_color = 'darkgreen' if improvement >= 0 else 'red'
											ax.annotate(
													f"{'+' if improvement >= 0 else ''}{improvement:.1f}%",
													xy=(k, finetuned_val),
													xytext=(5, 5),
													textcoords='offset points',
													fontsize=8.5,
													fontweight='bold',
													color=text_color,
													bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
											)
					# Axes formatting
					ax.set_xlabel('K', fontsize=11)
					ax.set_title(f'{metric}@K', fontsize=10, fontweight='bold')
					ax.grid(True, linestyle='--', alpha=0.9)
					ax.set_xticks(k_values)
					ax.set_ylim(-0.01, 1.01)  # Fixed range for consistency
					if j == 0:  # Add legend to first subplot only
							ax.legend(fontsize=9, loc='best')
					# Print detailed metrics to logs (optional, retained from original)
					if pretrained_values and finetuned_values:
							print(f"K-specific performance:")
							print(f"{'K':<5} {'Pre-trained':<12} {'Fine-tuned':<12} {'Improvement':<12}")
							for k, pre, fine in zip(k_values, pretrained_values, finetuned_values):
									imp = ((fine - pre) / pre * 100) if pre != 0 else 0
									print(f"{k:<5} {pre:.4f}{' ':<10} {fine:.4f}{' ':<10} {imp:+.2f}%")
							avg_improvement = sum(((fine - pre) / pre * 100 if pre != 0 else 0) for pre, fine in zip(pretrained_values, finetuned_values)) / len(k_values)
							print(f"\nAverage improvement: {avg_improvement:+.2f}%")
			# Save the figure for this mode
			plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
			print(f"Saved: {file_path}")
			plt.close(fig)
	# Print overall summary (simplified, retained from original)
	print(f"\n{'='*40}")
	print(f"OVERALL PERFORMANCE SUMMARY")
	print(f"{'='*40}")
	for mode in modes:
			pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
			finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
			print(f"\nMode: {mode}")
			for metric in metrics:
					k_values = sorted(
							k for k in topK_values if
							str(k) in pretrained_dict.get(model_name, {}).get(metric, {}) and
							str(k) in finetuned_dict.get(model_name, {}).get(finetune_strategy, {}).get(metric, {})
					)
					if k_values:
							improvements = [
									((finetuned_dict[model_name][finetune_strategy][metric][str(k)] - pretrained_dict[model_name][metric][str(k)]) /
									 pretrained_dict[model_name][metric][str(k)] * 100)
									for k in k_values if pretrained_dict[model_name][metric][str(k)] != 0
							]
							if improvements:
									avg_improvement = sum(improvements) / len(improvements)
									print(f"  {metric}: Average improvement across all K values: {avg_improvement:+.2f}%")
	print(f"\n{'='*80}")

def plot_comparison_metrics_merged_orig(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,  # e.g., 'ViT-B/32'
		finetune_strategy: str,  # e.g., 'LoRA'
		results_dir: str,
		topK_values: list,
		figure_size=(14, 5),
		DPI: int=300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	print(f"pretrained_img2txt_dict:")
	print(json.dumps(pretrained_img2txt_dict, indent=4, ensure_ascii=False))
	print(f"pretrained_txt2img_dict:")
	print(json.dumps(pretrained_txt2img_dict, indent=4, ensure_ascii=False))
	print()

	print(f"finetuned_img2txt_dict:")
	print(json.dumps(finetuned_img2txt_dict, indent=4, ensure_ascii=False))
	print(f"finetuned_txt2img_dict:")
	print(json.dumps(finetuned_txt2img_dict, indent=4, ensure_ascii=False))
	
	if model_name not in finetuned_img2txt_dict or finetune_strategy not in finetuned_img2txt_dict[model_name]:
		print(f"WARNING: {finetune_strategy} for {model_name} not found in finetuned_img2txt_dict. Skipping...")
		return
	if model_name not in finetuned_txt2img_dict or finetune_strategy not in finetuned_txt2img_dict[model_name]:
		print(f"WARNING: {finetune_strategy} for {model_name} not found in finetuned_txt2img_dict. Skipping...")
		return

	model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
	model_colors = plt.cm.tab10.colors
	
	# Print detailed analysis information to logs
	print(f"\n{'='*80}")
	print(f"DETAILED PERFORMANCE ANALYSIS FOR {dataset_name} - {model_name} WITH {finetune_strategy.capitalize()} FINE-TUNING")
	print(f"{'='*80}")
	
	# Create separate figures for each mode
	for i, mode in enumerate(modes):
		# Create a new figure for each mode
		fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
		fname = f"{dataset_name}_{finetune_strategy}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_retrieval_performance_comparison_{mode.replace('-', '_')}_merged.png"
		# Set a descriptive title for the figure
		file_path = os.path.join(results_dir, fname)
		fig.suptitle(
			f'$\\it{{{mode}}}$ Retrieval Performance Comparison\n'
			f'Pre-trained CLIP {model_name} vs. {finetune_strategy.capitalize()} Fine-tuning',
			fontsize=12,
			fontweight='bold',
		)
		
		# Select the appropriate dictionaries
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		
		print(f"\n{'-'*40}")
		print(f"MODE: {mode}: {fname}")
		print(f"{'-'*40}")
		
		for j, metric in enumerate(metrics):
			ax = axes[j]
			
			print(f"\nMetric: {metric}")
			print("-" * 20)
			
			# Lists to store values for logging
			k_values = []
			pretrained_values = []
			finetuned_values = []
			improvement_percentages = []
			
			# Plot pre-trained model performance
			if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
				k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
				values = [pretrained_dict[model_name][metric][str(k)] for k in k_values]
				pretrained_values = values
				
				pretrained_line, = ax.plot(
					k_values,
					values,
					label=f"Pre-trained CLIP {model_name}",
					color=model_colors[model_name_idx],
					marker='o',
					linestyle='--',
					linewidth=2,
					markersize=5,
					alpha=0.7,
				)
			
			# Plot fine-tuned model performance
			if (
				model_name in finetuned_dict and 
				metric in finetuned_dict[model_name][finetune_strategy] and
				finetune_strategy in finetuned_dict[model_name]
			):
				k_values = sorted([int(k) for k in finetuned_dict[model_name][finetune_strategy][metric].keys() if int(k) in topK_values])
				values = [finetuned_dict[model_name][finetune_strategy][metric][str(k)] for k in k_values]
				finetuned_values = values
				
				finetuned_line, = ax.plot(
					k_values,
					values,
					label=f"{finetune_strategy.capitalize()} Fine-tune",
					color=model_colors[model_name_idx],
					marker='s',
					linestyle='-',
					linewidth=2,
					markersize=5
				)
				
				# Add improvement percentages at key points
				if model_name in pretrained_dict:
					key_k_values = [1, 10, 20]  # Annotate these K values if available
					for k in key_k_values:
						if k in k_values:
							k_idx = k_values.index(k)
							pretrained_val = pretrained_dict[model_name][metric][str(k)]
							finetuned_val = values[k_idx]
							improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
							
							# Set color based on improvement value
							text_color = 'darkgreen' if improvement >= 0 else 'red'
							
							# Place annotations to the right with slight upward offset
							ax.annotate(
								f"{'+' if improvement >= 0 else ''}{improvement:.1f}%",
								xy=(k, finetuned_val),
								xytext=(5, 5),  # Fixed offset to the right and slightly up
								textcoords='offset points',
								fontsize=8.5,
								fontweight='bold',
								color=text_color,  # Apply the color
								bbox=dict(
									facecolor='white',
									edgecolor='none',
									alpha=0.7,
									pad=0.3
								)
							)
			
			ax.set_xlabel('K', fontsize=11)
			# ax.set_ylabel(f'{metric}@K', fontsize=11)
			ax.set_title(f'{metric}@K', fontsize=10, fontweight='bold')
			ax.grid(True, linestyle='--', alpha=0.9)
			ax.set_xticks(topK_values)
			
			# Set y-axis limits based on data
			all_values = []
			if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
				all_values.extend([pretrained_dict[model_name][metric][str(k)] for k in k_values if str(k) in pretrained_dict[model_name][metric]])
			if model_name in finetuned_dict and metric in finetuned_dict[model_name] and finetune_strategy in finetuned_dict[model_name]:
				all_values.extend([finetuned_dict[model_name][metric][str(k)] for k in k_values if str(k) in finetuned_dict[model_name][metric]])
			
			if all_values:
				min_val = min(all_values)
				max_val = max(all_values)
				padding = 0.1 * (max_val - min_val) if max_val > min_val else 0.1
				# ax.set_ylim(bottom=max(0, min_val - padding), top=min(1.0, max_val + padding))
				ax.set_ylim(bottom=-0.01, top=1.01)
			
			# Add legend to first subplot only
			if j == 0:
					ax.legend(fontsize=9, loc='best')
					
			# Print detailed metrics to logs
			if improvement_percentages:
					# Print K-specific improvements
					print(f"K-specific performance:")
					print(f"{'K':<5} {'Pre-trained':<12} {'Fine-tuned':<12} {'Improvement':<12}")
					for k, imp, pre, fine in improvement_percentages:
							print(f"{k:<5} {pre:.4f}{' ':<10} {fine:.4f}{' ':<10} {imp:+.2f}%")
					
					# Print summary statistics
					avg_improvement = sum([imp for _, imp, _, _ in improvement_percentages]) / len(improvement_percentages)
					max_improvement_data = max(improvement_percentages, key=lambda x: x[1])
					min_improvement_data = min(improvement_percentages, key=lambda x: x[1])
					
					print("\nSummary statistics:")
					print(f"Average improvement: {avg_improvement:+.2f}%")
					print(f"Maximum improvement: {max_improvement_data[1]:+.2f}% at K={max_improvement_data[0]}")
					print(f"Minimum improvement: {min_improvement_data[1]:+.2f}% at K={min_improvement_data[0]}")
					
					# Calculate Area Under the Curve improvement if applicable
					if len(pretrained_values) == len(finetuned_values) and len(pretrained_values) > 1:
							pre_auc = sum(pretrained_values)
							fine_auc = sum(finetuned_values)
							auc_improvement = ((fine_auc - pre_auc) / pre_auc) * 100
							print(f"Area Under Curve improvement: {auc_improvement:+.2f}%")
		
		# Save the figure for this mode
		plt.savefig(
			fname=file_path,
			dpi=DPI,
			bbox_inches='tight',
		)
		plt.close(fig)
	
	# Print overall summary
	print(f"\n{'='*40}")
	print(f"OVERALL PERFORMANCE SUMMARY")
	print(f"{'='*40}")
	
	for mode in modes:
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		
		print(f"\nMode: {mode}")
		for metric in metrics:
			if (model_name in pretrained_dict and metric in pretrained_dict[model_name] and
				model_name in finetuned_dict and metric in finetuned_dict[model_name]):
				
				# Calculate average improvement across all K values
				k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
				improvements = []
				
				for k in k_values:
					str_k = str(k)
					if str_k in pretrained_dict[model_name][metric] and str_k in finetuned_dict[model_name][metric]:
						pretrained_val = pretrained_dict[model_name][metric][str_k]
						finetuned_val = finetuned_dict[model_name][metric][str_k]
						improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
						improvements.append(improvement)
				
				if improvements:
					avg_improvement = sum(improvements) / len(improvements)
					print(f"  {metric}: Average improvement across all K values: {avg_improvement:+.2f}%")
	
	print(f"\n{'='*80}")

def plot_comparison_metrics_detailed(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,  # e.g., 'ViT-B/32'
		finetune_strategy: str,  # e.g., 'LoRA'
		topK_values: list,
		fname_prefix: str="Comparison_Metrics",
		fname: str="comparison.png",
		figure_size=(15, 8),
		DPI: int=300,
	):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
		model_colors = plt.cm.tab10.colors
		
		# Create separate figures for each mode
		for i, mode in enumerate(modes):
				# Create a new figure for each mode
				fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
				
				# Set a descriptive title for the figure
				fig.suptitle(
						f"{dataset_name} Dataset - {mode} Retrieval Performance\n"
						f"Pre-trained CLIP {model_name} vs. {finetune_strategy.capitalize()} Fine-tuning",
						fontsize=14,
						fontweight='bold',
				)
				
				# Select the appropriate dictionaries
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				
				# Track best improvement for annotation
				best_improvements = {metric: {'value': 0, 'k': 0} for metric in metrics}
				
				for j, metric in enumerate(metrics):
						ax = axes[j]
						
						# Create lists to store values for statistical annotations
						pretrained_values = []
						finetuned_values = []
						
						# Plot pre-trained model performance
						if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
								k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
								values = [pretrained_dict[model_name][metric][str(k)] for k in k_values]
								pretrained_values = values
								
								pretrained_line, = ax.plot(
										k_values,
										values,
										label=f"Pre-trained",
										color=model_colors[model_name_idx],
										marker='o',
										linestyle='--',
										linewidth=2,
										markersize=5,
										alpha=0.7,
								)
						
						# Plot fine-tuned model performance
						if model_name in finetuned_dict and metric in finetuned_dict[model_name]:
								k_values = sorted([int(k) for k in finetuned_dict[model_name][metric].keys() if int(k) in topK_values])
								values = [finetuned_dict[model_name][metric][str(k)] for k in k_values]
								finetuned_values = values
								
								finetuned_line, = ax.plot(
										k_values,
										values,
										label=f"{finetune_strategy} Fine-tuned",
										color=model_colors[model_name_idx],
										marker='s',
										linestyle='-',
										linewidth=2,
										markersize=5
								)
								
								# Add improvement percentages at key points
								if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
										# Calculate improvement at all K values
										all_improvements = []
										for k_idx, k in enumerate(k_values):
												if str(k) in pretrained_dict[model_name][metric]:
														pretrained_val = pretrained_dict[model_name][metric][str(k)]
														finetuned_val = values[k_idx]
														improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
														all_improvements.append((k, improvement))
														
														# Track the best improvement
														if improvement > best_improvements[metric]['value']:
																best_improvements[metric]['value'] = improvement
																best_improvements[metric]['k'] = k
										
										# Annotate key improvement points (top 3 improvements)
										sorted_improvements = sorted(all_improvements, key=lambda x: abs(x[1]), reverse=True)
										for idx, (k, improvement) in enumerate(sorted_improvements[:2]):  # Annotate top 2 improvements
												k_idx = k_values.index(k)
												finetuned_val = values[k_idx]
												
												# Set color based on improvement value
												text_color = 'darkgreen' if improvement >= 0 else 'red'
												
												# Place annotations to the right with slight upward offset
												ax.annotate(
														f"{'+' if improvement >= 0 else ''}{improvement:.1f}% @K={k}",
														xy=(k, finetuned_val),
														xytext=(5, 5 + idx * 15),  # Offset to avoid overlap
														textcoords='offset points',
														fontsize=8.5,
														fontweight='bold',
														color=text_color,
														bbox=dict(
																facecolor='white',
																edgecolor='none',
																alpha=0.7,
																pad=0.3
														),
														arrowprops=dict(
																arrowstyle="->",
																color=text_color,
																shrinkA=5,
																shrinkB=5,
																alpha=0.7
														)
												)
						
						# Add statistical analysis summary if we have both pretrained and finetuned values
						if pretrained_values and finetuned_values and len(pretrained_values) == len(finetuned_values):
								# Calculate average improvement
								avg_improvement = sum([((f - p) / p) * 100 for p, f in zip(pretrained_values, finetuned_values)]) / len(pretrained_values)
								
								# Calculate maximum improvement
								max_improvement = max([((f - p) / p) * 100 for p, f in zip(pretrained_values, finetuned_values)])
								max_k = k_values[np.argmax([((f - p) / p) * 100 for p, f in zip(pretrained_values, finetuned_values)])]
								
								# Add text box with statistics
								stat_text = (
										f"Average Improvement: {avg_improvement:.1f}%\n"
										f"Maximum Improvement: {max_improvement:.1f}% @K={max_k}"
								)
								
								# Add the statistics box in upper left or right corner
								x_pos = 0.05 if avg_improvement > 0 else 0.55  # Left if positive, right if negative
								ax.text(
										x_pos, 0.95, stat_text,
										transform=ax.transAxes,
										fontsize=8,
										verticalalignment='top',
										bbox=dict(
												boxstyle='round,pad=0.5',
												facecolor='white',
												alpha=0.8,
												edgecolor='gray'
										)
								)
						
						# Configure axes
						ax.set_xlabel('K', fontsize=12)
						ax.set_ylabel(f'{metric}@K', fontsize=12)
						ax.set_title(f'{metric}@K', fontsize=14)
						ax.grid(True, linestyle='--', alpha=0.7)
						ax.set_xticks(topK_values)
						
						# Set y-axis limits based on data
						all_values = []
						if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
								all_values.extend([pretrained_dict[model_name][metric][str(k)] for k in k_values if str(k) in pretrained_dict[model_name][metric]])
						if model_name in finetuned_dict and metric in finetuned_dict[model_name]:
								all_values.extend([finetuned_dict[model_name][metric][str(k)] for k in k_values if str(k) in finetuned_dict[model_name][metric]])
						
						if all_values:
								min_val = min(all_values)
								max_val = max(all_values)
								padding = 0.1 * (max_val - min_val) if max_val > min_val else 0.1
								ax.set_ylim(bottom=max(0, min_val - padding), top=min(1.0, max_val + padding))
						
						# Add legend to first subplot only
						if j == 0:
								ax.legend(fontsize=10, loc='lower right')
				
				# Add a summary of best improvements across metrics at the bottom of the figure
				summary = "\n".join([
						f"Best {metric} improvement: {best_improvements[metric]['value']:.1f}% at K={best_improvements[metric]['k']}"
						for metric in metrics
				])
				fig.text(
						0.5, 0.01, 
						summary,
						ha='center',
						fontsize=9,
						bbox=dict(
								boxstyle='round,pad=0.5',
								facecolor='lightyellow',
								alpha=0.8,
								edgecolor='gray'
						)
				)
				
				# Save the figure for this mode
				plt.savefig(fname=f"{fname_prefix}_{mode.replace('-', '_')}.png", dpi=DPI, bbox_inches='tight')
				plt.close(fig)
				
		# Create an additional summary plot showing the relative percentage improvements
		fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
		fig.suptitle(
				f"{dataset_name} Dataset - Relative Improvement from Fine-tuning\n"
				f"CLIP {model_name} with {finetune_strategy.capitalize()} Strategy",
				fontsize=14,
				fontweight='bold',
		)
		
		for j, metric in enumerate(metrics):
				ax = axes[j]
				
				# Extract improvement percentages for both modes
				improvements_by_mode = {}
				
				for mode in modes:
						pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
						finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
						
						if (model_name in pretrained_dict and metric in pretrained_dict[model_name] and
								model_name in finetuned_dict and metric in finetuned_dict[model_name]):
								
								k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
								improvements = []
								
								for k in k_values:
										str_k = str(k)
										if str_k in pretrained_dict[model_name][metric] and str_k in finetuned_dict[model_name][metric]:
												pretrained_val = pretrained_dict[model_name][metric][str_k]
												finetuned_val = finetuned_dict[model_name][metric][str_k]
												improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
												improvements.append(improvement)
								
								improvements_by_mode[mode] = {
										'k_values': k_values,
										'improvements': improvements
								}
				
				# Plot improvements for each mode
				bar_width = 0.35
				for mode_idx, mode in enumerate(modes):
						if mode in improvements_by_mode:
								k_values = improvements_by_mode[mode]['k_values']
								improvements = improvements_by_mode[mode]['improvements']
								
								x_positions = np.array(range(len(k_values))) + (mode_idx - 0.5) * bar_width
								bars = ax.bar(
										x_positions,
										improvements,
										bar_width,
										label=mode,
										color=model_colors[mode_idx],
										alpha=0.7
								)
								
								# Add value labels on top of bars
								for bar_idx, bar in enumerate(bars):
										height = bar.get_height()
										align = 'center'
										va = 'bottom' if height >= 0 else 'top'
										
										ax.text(
												bar.get_x() + bar.get_width() / 2,
												height + (5 if height >= 0 else -5),
												f"{height:.1f}%",
												ha=align,
												va=va,
												fontsize=8,
												rotation=90,
												color='black'
										)
				
				# Configure axes
				ax.set_xlabel('K', fontsize=12)
				ax.set_ylabel('Relative Improvement (%)', fontsize=12)
				ax.set_title(f'{metric}@K Improvement', fontsize=14)
				ax.grid(True, linestyle='--', alpha=0.7, axis='y')
				
				# Set x-axis ticks to K values
				if any(improvements_by_mode.values()):
						# Use first available mode's K values for x-axis labels
						first_mode = next(iter(improvements_by_mode.values()))
						ax.set_xticks(range(len(first_mode['k_values'])))
						ax.set_xticklabels([f"K={k}" for k in first_mode['k_values']])
				
				# Add zero line for reference
				ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
				
				# Add legend to first subplot only
				if j == 0:
						ax.legend(fontsize=10)
		
		# Save the improvement summary figure
		plt.savefig(
			fname=f"{fname_prefix}_Relative_Improvements.png",
			dpi=DPI, 
			bbox_inches='tight',
		)
		plt.close(fig)



class EarlyStopping:
	def __init__(
			self,
			patience: int = 5, # epochs to wait before stopping the training
			min_delta: float = 1e-3, # minimum difference between new and old loss to count as improvement
			cumulative_delta: float = 0.01,
			window_size: int = 5,
			mode: str = 'min',
			min_epochs: int = 5,
			restore_best_weights: bool = True,
		):
		"""
		Args:
			patience: Number of epochs to wait before early stopping
			min_delta: Minimum change in monitored value to qualify as an improvement
			cumulative_delta: Minimum cumulative improvement over window_size epochs
			window_size: Size of the window for tracking improvement trends
			mode: 'min' for loss, 'max' for metrics like accuracy
			min_epochs: Minimum number of epochs before early stopping can trigger
			restore_best_weights: Whether to restore model to best weights when stopped
		"""

		# Validate inputs
		if mode not in ["min", "max"]:
			raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")
		if window_size < 0:
			raise ValueError(f"Invalid window_size: {window_size}. Must be ≥ 0.")

		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights

		self.sign = 1 if mode == 'min' else -1
		self.reset()
		
	def reset(self):
		self.best_score = None
		self.best_weights = None
		self.counter = 0
		self.stopped_epoch = 0
		self.value_history = []
		self.improvement_history = []
		self.best_epoch = 0

	def is_improvement(
			self,
			current_value: float,
		) -> bool:
		if self.best_score is None:
			return True
		improvement = (self.best_score - current_value) * self.sign
		improved = improvement > self.min_delta
		print(f"\tImprovement(w.r.t. best score={self.best_score}): {improvement} > min_delta: {self.min_delta}: {improved}")
		return improved
	
	def calculate_trend(self) -> float:
		"""
			Calculate improvement trend over window.
			Returns inf (mode='min') or -inf (mode='max') if history is shorter than window_size,
			effectively disabling window-based stopping until enough epochs have passed.
		"""
		if self.window_size == 0:
			return float("inf") if self.mode == "min" else float("-inf")

		if len(self.value_history) < self.window_size:
			return float('inf') if self.mode == 'min' else float('-inf')

		window = self.value_history[-self.window_size:]

		# # Calculate the trend over the window:
		# if self.mode == 'min':
		# 	return sum(window[i] - window[i+1] for i in range(len(window)-1))
		# return sum(window[i+1] - window[i] for i in range(len(window)-1))

		# simple trend calculation:
		if self.mode == 'min':
			return window[0] - window[-1]  # Total improvement over the window
		else:
			return window[-1] - window[0] # For accuracy-like metrics, we want to see the increase
	
	def should_stop(
			self,
			current_value: float, 
			model: torch.nn.Module, 
			epoch: int,
		) -> bool:
		"""
		Enhanced stopping decision based on multiple criteria.
		
		Args:
				current_value: Current value of the monitored metric (e.g., validation loss).
				model: The model being trained.
				epoch: Current epoch number.
		
		Returns:
				bool: Whether to stop training.
		"""
		self.value_history.append(current_value)
		if epoch < self.min_epochs:
			print(f"Epoch {epoch+1}: Still less than minimum epochs. Skipping early stopping (min_epochs={self.min_epochs})")
			return False
		
		if self.is_improvement(current_value):
			print(f"\tImprovement detected (current: {current_value}, best: {self.best_score})")
			self.best_score = current_value
			self.stopped_epoch = epoch
			self.best_epoch = epoch
			if self.restore_best_weights:
				self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
			self.counter = 0
			self.improvement_history.append(True)
		else:
			print(
				f"\tNO improvement detected! (current={current_value}, best={self.best_score}) "
				f"absolute difference: {abs(current_value - self.best_score)} "
				f"=> Incrementing counter (current counter={self.counter})"
			)
			self.counter += 1
			self.improvement_history.append(False)
		
		trend = self.calculate_trend()
		cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		print(f">> Trend: {trend} | Cumulative Improvement: {cumulative_improvement} (cumulative_delta={self.cumulative_delta})")
		
		should_stop = False
		if self.counter >= self.patience:
			print(f"Early stopping can be triggered since validation loss fails to improve for (patience={self.patience}) epochs.")
			should_stop = True
		
		if len(self.improvement_history) >= self.window_size:
			recent_improvements = sum(self.improvement_history[-self.window_size:])
			if recent_improvements == 0 and cumulative_improvement < self.cumulative_delta:
				print("Early stopping triggered (local optimum detected).")
				should_stop = True
		
		if should_stop and self.restore_best_weights and self.best_weights is not None:
			model.load_state_dict(self.best_weights)
			print("Restored best model weights.")
		
		return should_stop

	def get_best_score(self) -> float:
		return self.best_score
	
	def get_stopped_epoch(self) -> int:
		return self.stopped_epoch

	def get_best_epoch(self) -> int:
		return self.best_epoch

def should_transition_phase(
		losses: list[float],
		window: int,
		best_loss: float,
		best_loss_threshold: float = 1e-3,
		volatility_threshold: float = 10.0,  	# High volatility threshold (CV%)
		slope_threshold: float = 0.0,        	# Slope must be > this to indicate increase
		pairwise_imp_threshold: float = 5e-3  # Minimum pairwise improvement
	) -> bool:
	
	if len(losses) < window:
		print(f"<!> Not enough loss data ({len(losses)} < {window} epochs) to evaluate phase transition.")
		return False
	
	last_window = losses[-window:]
	current_loss = last_window[-1]

	# --- METRIC CALCULATIONS ---
	# 1. Volatility (Coefficient of Variation)
	mean_loss = np.mean(last_window)
	std_loss = np.std(last_window)
	cv = (std_loss / mean_loss) * 100 if mean_loss != 0 else 0  # Volatility in %

	# 2. Pairwise Improvement (average improvement per step)
	pairwise_diffs = [last_window[i] - last_window[i+1] for i in range(len(last_window)-1)]
	pairwise_imp_avg = np.mean(pairwise_diffs) if pairwise_diffs else 0

	# 3. Trend Slope (using linear regression)
	def compute_slope(losses):
			x = np.arange(len(losses))
			A = np.vstack([x, np.ones(len(x))]).T
			m, c = np.linalg.lstsq(A, losses, rcond=None)[0]
			return m  # Slope
	
	slope = compute_slope(last_window)

	# 4. Proximity to best loss
	close_to_best = abs(current_loss - best_loss) < best_loss_threshold

	# --- DECISION LOGIC ---
	transition = False
	reasons = []

	# Check volatility
	if cv >= volatility_threshold:
			transition = True
			reasons.append(f"High volatility (CV={cv:.3f}%)")

	# Check upward trend (loss increasing)
	if slope > slope_threshold:
			transition = True
			reasons.append(f"Positive slope (slope={slope})")

	# Check insufficient improvement and not near best loss
	if (pairwise_imp_avg < pairwise_imp_threshold) and (not close_to_best):
			transition = True
			reasons.append(f"Low improvement ({pairwise_imp_avg} < {pairwise_imp_threshold})")

	# --- DEBUGGING OUTPUTS ---
	print(f"\nPhase Transition Evaluation (Window={window}):")
	print(f"Last {window} Losses:\n{last_window}")
	print(f"\tCurrent Loss: {current_loss} | Best Loss: {best_loss}")
	print(f"\tVolatility (CV%): {cv:.3f}% | Pairwise Improvement: {pairwise_imp_avg}")
	print(f"\tTrend Slope: {slope} (Positive = increasing loss)")
	print(f"\tClose to best loss? {close_to_best}")

	if transition:
		print(f"==>> Transition Required! Reasons: {', '.join(reasons)}")
	else:
		print("==>> No transition needed: Stable and improving.")
	print("-"*100)
	return transition

def handle_phase_transition(
		current_phase: int,
		initial_lr: float,
		max_phases: int,
		scheduler,
		window_size: int,  # Add window_size parameter
		current_loss: float,  # Add current validation loss
		best_loss: float,  # Add best observed loss
	):

	# Calculate adaptive LR reduction factor
	loss_ratio = current_loss / best_loss if best_loss > 0 else 1.0
	window_factor = max(0.5, min(2.0, 10/window_size))  # 0.5-2.0 range
	
	if current_phase >= max_phases - 1:
		# Final phase uses dynamic minimum LR
		new_lr = initial_lr * (0.1 * window_factor)
		print(f"Final phase LR: {new_lr:.2e} (window factor: {window_factor})")
		return current_phase, new_lr
	
	# Dynamic phase-based reduction with window consideration
	phase_factor = 0.8 ** (current_phase / (window_size/5))  # Scaled by window
	new_lr = initial_lr * max(0.05, phase_factor * loss_ratio)  # Never <5% of initial
	
	# Update scheduler
	if isinstance(scheduler, lr_scheduler.OneCycleLR):
		scheduler.max_lr = new_lr
		scheduler.base_lrs = [new_lr] * len(scheduler.base_lrs)
	
	for param_group in scheduler.optimizer.param_groups:
		param_group['lr'] = new_lr
	
	print(
		f"Phase {current_phase+1} LR: {new_lr:.2e} | "
		f"Factors: phase={phase_factor}, window={window_factor}, "
		f"loss={loss_ratio}"
	)	
	return current_phase + 1, new_lr

def progressive_unfreeze_finetune(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		nw: int,
		print_every: int,
		initial_learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int,
		patience: int = 10,
		min_delta: float = 1e-3,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		top_k_values: List[int] = [1, 5, 10, 15, 20],
		layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
	) -> Dict[str, any]:

	# Input validation
	if not train_loader or not validation_loader:
		raise ValueError("Train and validation loaders must not be empty.")
	
	# Initialize early stopping
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
	)
	
	# Extract dataset name
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = getattr(validation_loader.dataset, 'dataset_name', 'Unknown')
	
	mode = inspect.stack()[0].function
	model_arch = model.name
	model_name = model.__class__.__name__
	
	# just for debugging:
	# for name, param in model.named_parameters():
	# 	print(f"{name} => {param.shape} {param.requires_grad}")
	
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))

	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))
	
	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break
	
	unfreeze_percentages = get_unfreeze_pcts_hybrid(
		model=model,
		train_loader=train_loader,
		min_phases=7,
		max_phases=15,
	)

	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		unfreeze_percentages=unfreeze_percentages,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"dropout_{dropout_val}_init_lr_{initial_learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
	)
	
	criterion = torch.nn.CrossEntropyLoss()

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	optimizer = AdamW(
		params=filter(lambda p: p.requires_grad, model.parameters()),
		lr=initial_learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	scheduler = lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=initial_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	training_losses = []
	metrics_for_all_epochs = []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	global_patience = int(minimum_epochs * 1.5) # Total epochs without improvement across phases
	global_counter = 0
	global_best_loss = float('inf')
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None
	current_phase = 0
	epochs_in_current_phase = 0
	min_epochs_per_phase = 5

	# global minimum number of epochs that must be completed 
	# before any phase transition can occur:
	min_epochs_before_transition = min(window_size, minimum_epochs)

	min_phases_before_stopping = 3 # ensure model progresses through at least 3 phases (unfreezing 60% of transformer blocks) before early stopping can trigger
	layer_cache = {} # Cache for layer freezing status
	learning_rate = None

	train_start_time = time.time()
	for epoch in range(num_epochs):
		torch.cuda.empty_cache()
		epochs_in_current_phase += 1 # Increment the epoch counter for the current phase
		print(f"Epoch [{epoch+1}/{num_epochs}] GPU Memory usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
		print(
			f"epoch: {epoch} min_epochs_before_transition: {min_epochs_before_transition} "
			f"epoch >= min_epochs_before_transition: {epoch >= min_epochs_before_transition}\n"
			f"epochs_in_current_phase: {epochs_in_current_phase} min_epochs_per_phase: {min_epochs_per_phase} "
			f"epochs_in_current_phase >= min_epochs_per_phase: {epochs_in_current_phase >= min_epochs_per_phase}\n"
			f"epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase: "
			f"{epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase}"
		)

		# Phase transition logic
		if epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase:
			img2txt_accs = [metrics["img2txt_acc"] for metrics in metrics_for_all_epochs]
			txt2img_accs = [metrics["txt2img_acc"] for metrics in metrics_for_all_epochs]
			avg_accs = [(img + txt) / 2 for img, txt in zip(img2txt_accs, txt2img_accs)]

			# should_transition = should_transition_phase(
			# 	losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
			# 	accuracies=None,#avg_accs,
			# 	loss_threshold=min_delta,
			# 	accuracy_threshold=5e-5,
			# 	best_loss_threshold=1e-3,
			# 	window=window_size,
			# 	best_loss=best_val_loss,
			# )

			should_transition = should_transition_phase(
				losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
				window=window_size,
				best_loss=best_val_loss,
				best_loss_threshold=1e-3,
			)

			if should_transition:
				current_phase, learning_rate = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=len(unfreeze_schedule),
					scheduler=scheduler,
					window_size=window_size,
					current_loss=metrics_for_all_epochs[-1]["val_loss"],
					best_loss=best_val_loss,
				)
				epochs_in_current_phase = 0  # Reset the counter after transitioning
				early_stopping.reset() # Reset early stopping after phase transition (between phases)

				# Update optimizer with new learning rate
				for param_group in optimizer.param_groups:
					param_group['lr'] = learning_rate

				scheduler.base_lrs = [learning_rate] * len(scheduler.base_lrs)
				scheduler.max_lr = learning_rate

		# Unfreeze layers for current phase
		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)
		
		model.train()
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True) # Clear gradients at start of each epoch

			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=True): # Automatic Mixed Precision (AMP) backpropagation
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)

			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"Batch [{bidx+1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)
		
		# Evaluate on validation set
		metrics_per_epoch = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=top_k_values,
		)
		metrics_for_all_epochs.append(metrics_per_epoch)
		
		# Compute retrieval metrics
		img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=top_k_values,
		)

		img2txt_metrics_list.append(img2txt_metrics)
		txt2img_metrics_list.append(txt2img_metrics)
		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
			f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
			f'[image retrieval per text]: {metrics_per_epoch.get("txt2img_acc")}'
		)

		# Checkpointing
		current_val_loss = metrics_per_epoch["val_loss"]
		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_loss": best_val_loss,
		}
		if current_val_loss < best_val_loss - early_stopping.min_delta:
			# print(f"New best model found: (current loss: {current_val_loss} < best loss: {best_val_loss})")
			best_val_loss = current_val_loss
			checkpoint.update({"best_val_loss": best_val_loss})
			torch.save(checkpoint, mdl_fpth)
			best_img2txt_metrics = img2txt_metrics
			best_txt2img_metrics = txt2img_metrics

		# Early stopping (per-phase)
		if early_stopping.should_stop(current_val_loss, model, epoch):
			if current_phase >= min_phases_before_stopping:
				print(f"[Per Phase] Early stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score()}")
				break
			else:
				print(
					f"[Per Phase] Early stopping condition met at epoch {epoch + 1}! "
					f"but delaying until minimum phases ({min_phases_before_stopping}) are reached. "
					f"Current phase: {current_phase}"
				)
		
		# Global Early Stopping
		if current_val_loss < global_best_loss - min_delta:
			global_best_loss = current_val_loss
			global_counter = 0
		else:
			global_counter += 1
		if global_counter >= global_patience and current_phase >= min_phases_before_stopping:
			print(f"Global early stopping triggered after (global_patience={global_patience}) epochs without improvement.")
			break

		print("-" * 140)

	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	file_base_name = (
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"ep_{len(training_losses)}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"dropout_{dropout_val}_"
		f"init_lr_{initial_learning_rate:.1e}"
	)

	if learning_rate is not None:
		file_base_name += f"_final_lr_{learning_rate:.1e}"


	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"val_acc": os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png"),
		"img2txt_topk": os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png"),
		"txt2img_topk": os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	plot_loss_accuracy(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
		val_acc_img2txt_list=[metrics["img2txt_acc"] for metrics in metrics_for_all_epochs],
		val_acc_txt2img_list=[metrics["txt2img_acc"] for metrics in metrics_for_all_epochs],
		img2txt_topk_accuracy_list=[metrics["img2txt_topk_acc"] for metrics in metrics_for_all_epochs],
		txt2img_topk_accuracy_list=[metrics["txt2img_topk_acc"] for metrics in metrics_for_all_epochs],
		mean_reciprocal_rank_list=[metrics["mean_reciprocal_rank"] for metrics in metrics_for_all_epochs],
		cosine_similarity_list=[metrics["cosine_similarity"] for metrics in metrics_for_all_epochs],
		losses_file_path=plot_paths["losses"],
		accuracy_file_path=plot_paths["val_acc"],
		img2txt_topk_accuracy_file_path=plot_paths["img2txt_topk"],
		txt2img_topk_accuracy_file_path=plot_paths["txt2img_topk"],
		mean_reciprocal_rank_file_path=plot_paths["mrr"],
		cosine_similarity_file_path=plot_paths["cs"],
	)

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

############################################ 27.03.2025 ############################################
def should_transition_phase_original(
	losses: List[float],
	accuracies: List[float] = None,
	loss_threshold: float = 5e-3,
	accuracy_threshold: float = 1e-3,
	best_loss_threshold: float = 1e-3,
	window: int = 10,
	best_loss: Optional[float] = None,
) -> bool:
"""
Determines if a phase transition is needed based on loss and accuracy trends.

Args:
	losses: List of training losses per epoch.
	accuracies: Optional list of validation accuracies per epoch.
	loss_threshold: Minimum cumulative loss improvement to avoid plateau.
	accuracy_threshold: Minimum cumulative accuracy improvement to avoid plateau.
	best_loss_threshold: Threshold for closeness to best loss.
	window: Number of epochs to evaluate trends over.
	best_loss: Optional best loss achieved so far.

Returns:
	bool: True if phase transition is required, False otherwise.
"""

if len(losses) < window:
	print(f"<!> Not enough loss data ({len(losses)} < {window} windows) to evaluate phase transition.")
	return False

# Loss analysis
last_window_losses = losses[-window:]

last_window_losses_mean = np.mean(last_window_losses)
last_window_losses_std = np.std(last_window_losses)
last_window_losses_cv = (last_window_losses_std / last_window_losses_mean) * 100 if last_window_losses_mean != 0 else 0

cumulative_loss_improvement = last_window_losses[0] - last_window_losses[-1]  # Positive = improvement
loss_trend = last_window_losses[-1] - last_window_losses[0]  # Positive = worsening
close_to_best = best_loss is not None and abs(last_window_losses[-1] - best_loss) < best_loss_threshold
loss_plateau = abs(cumulative_loss_improvement) < loss_threshold
sustained_improvement = cumulative_loss_improvement > loss_threshold  # Significant continuous improvement

pairwise_improvements = [last_window_losses[i] - last_window_losses[i+1] for i in range(len(last_window_losses)-1)]
average_improvement = np.mean(pairwise_improvements) if pairwise_improvements else 0.0 # positive → improvement

# Accuracy analysis
acc_plateau = False
cumulative_acc_improvement = None
if accuracies is not None and len(accuracies) >= window:
	last_window_accs = accuracies[-window:]
	cumulative_acc_improvement = last_window_accs[-1] - last_window_accs[0]  # Positive = improvement
	acc_plateau = abs(cumulative_acc_improvement) < accuracy_threshold

# Detailed debugging prints
print(f"\nPhase transition analysis over {window} Window Losses:")
print(f"Window Losses[{len(last_window_losses)}]/total losses[{len(losses)}]: Coefficient of Variation: {last_window_losses_cv:.3f}%")
print(last_window_losses)
print(
	f"\t|Cumulative loss improvement| = {abs(cumulative_loss_improvement)} "
	f"=> Loss plateau (<{loss_threshold}): {loss_plateau}"
)
print(
	f"\tCumulative loss improvement(first-last) = {cumulative_loss_improvement} "
	f"=> Sustained Improvement (>{loss_threshold}): {sustained_improvement}"
)
print(f"\tLoss trend(last-first): {loss_trend} (>0 worsening: {loss_trend > 0})")
print(
	f"\tCurrent loss: {last_window_losses[-1]} best loss: {best_loss if best_loss is not None else 'N/A'} | "
	f"Close to best loss (absolute diff) [<{best_loss_threshold}]: {close_to_best} "
)
print(f"pairwise_improvements[{len(pairwise_improvements)}]:\n{pairwise_improvements}")
print(f"\tAverage pairwise lost improvement (>0 improvement): {average_improvement}")

if accuracies is not None and len(accuracies) >= window:
	print(f"\t{window} Window accuracies: {last_window_accs}")
	print(f"\tCumulative accuracy improvement: {cumulative_acc_improvement} (threshold: {accuracy_threshold})")
	print(f"\tAccuracy plateau: {acc_plateau}")
# else:
# 	print("\t<!> Accuracy data unavailable; relying solely on loss for phase transition.")

transition = False

# Transition logic
if loss_plateau:
	if loss_trend > 0:
		transition = True
		print(f"\t>> Decision: Transition due to active loss deterioration, regardless of proximity to best loss! (loss_trend: {loss_trend} > 0)")
	elif close_to_best is False and sustained_improvement is False:
		transition = True
		print("\t>> Decision: Transition due to stagnation without proximity to best loss")
	else:
		print(
			f"\t>> Decision: No transition! Close to best loss ({close_to_best}) | Sustained improvement ({sustained_improvement})"
			f"For transition: Close to best loss must be False and sustained improvement must be False."
		)
elif acc_plateau:
	transition = True
	print("\t>> Decision: Transition due to accuracy plateau")

print(f"==>> Phase Transition Required? {transition}\n")
return transition


def progressive_unfreeze_finetune_micro_batch(
	model: torch.nn.Module,
	train_loader: DataLoader,
	validation_loader: DataLoader,
	num_epochs: int,
	nw: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	patience: int = 10,
	min_delta: float = 1e-4,
	cumulative_delta: float = 5e-3,
	minimum_epochs: int = 20,
	top_k_values: List[int] = [1, 5, 10, 15, 20],
	layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
	min_epochs_before_transition: int = 5,
	accumulation_steps: int = 4, # Gradient accumulation to control the number of micro-batches 
) -> Dict[str, any]:

if not train_loader or not validation_loader:
	raise ValueError("Train and validation loaders must not be empty.")

window_size = max(5, int(0.1 * len(train_loader)))  # 10% of training batches

print(f"Training batch: {len(train_loader)}, window_size: {window_size}")

early_stopping = EarlyStopping(
	patience=patience,
	min_delta=min_delta,
	cumulative_delta=cumulative_delta,
	window_size=window_size,
	mode='min',
	min_epochs=minimum_epochs,
	restore_best_weights=True,
)

try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
except AttributeError:
		dataset_name = getattr(validation_loader.dataset, 'dataset_name', 'Unknown')

# Create results directory
os.makedirs(results_dir, exist_ok=True)
mode = inspect.stack()[0].function
model_arch = model.name
model_name = model.__class__.__name__

print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))
if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

# Find dropout value
dropout_val = 0.0
for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
				dropout_val = module.p
				break

unfreeze_percentages = get_unfreeze_pcts_hybrid(
		model=model,
		train_loader=train_loader,
		min_phases=7,
		max_phases=15,
)
unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		unfreeze_percentages=unfreeze_percentages,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
)
mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"dropout_{dropout_val}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
)

# Initialize training components
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
)
optimizer = AdamW(
		params=filter(lambda p: p.requires_grad, model.parameters()),
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-8,
		weight_decay=weight_decay,
)
scheduler = lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
)

training_losses = []
metrics_for_all_epochs = []
img2txt_metrics_list = []
txt2img_metrics_list = []
train_start_time = time.time()
best_val_loss = float('inf')
best_img2txt_metrics = None
best_txt2img_metrics = None
current_phase = 0
epochs_in_current_phase = 0
min_epochs_per_phase = 5
max_epochs_per_phase = 15
initial_learning_rate = learning_rate
min_phases_before_stopping = 3  # Ensure model progresses through at least 3 phases before early stopping
layer_cache = {}  # Cache for layer freezing status
# Effective batch size and micro-batch size
effective_batch_size = train_loader.batch_size
micro_batch_size = effective_batch_size // accumulation_steps
if micro_batch_size == 0:
		micro_batch_size = 1
		accumulation_steps = effective_batch_size  # Adjust accumulation steps if batch size is too small
print(f"Effective Batch Size: {effective_batch_size}, Micro-Batch Size: {micro_batch_size}, Accumulation Steps: {accumulation_steps}")
for epoch in range(num_epochs):
		torch.cuda.empty_cache()
		print(f"Epoch [{epoch+1}/{num_epochs}] GPU Memory usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
		epochs_in_current_phase += 1
		# Phase transition logic
		if epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase:
				img2txt_accs = [metrics["img2txt_acc"] for metrics in metrics_for_all_epochs]
				txt2img_accs = [metrics["txt2img_acc"] for metrics in metrics_for_all_epochs]
				avg_accs = [(img + txt) / 2 for img, txt in zip(img2txt_accs, txt2img_accs)]
				should_transition = should_transition_phase(
						losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
						accuracies=avg_accs,
						loss_threshold=min_delta * 2,
						accuracy_threshold=1e-4,
						best_loss_threshold=min_delta * 5,
						window=window_size,
						best_loss=best_val_loss,
				)
				if should_transition:
						current_phase, learning_rate = handle_phase_transition(
								current_phase=current_phase,
								initial_lr=initial_learning_rate,
								max_phases=len(unfreeze_schedule),
								scheduler=scheduler,
						)
						epochs_in_current_phase = 0  # Reset the counter after transitioning
		
		# Unfreeze layers for current phase
		unfreeze_layers(
				model=model,
				strategy=unfreeze_schedule,
				phase=current_phase,
				cache=layer_cache,
		)
		
		# Update optimizer with new learning rate
		for param_group in optimizer.param_groups:
				param_group['lr'] = learning_rate
		
		model.train()
		epoch_loss = 0.0
		optimizer.zero_grad(set_to_none=True)  # Clear gradients at the start of the epoch
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
				# Split the batch into micro-batches
				num_samples = images.size(0)
				micro_batches = []
				for micro_idx in range(0, num_samples, micro_batch_size):
						micro_end = min(micro_idx + micro_batch_size, num_samples)
						micro_images = images[micro_idx:micro_end].to(device, non_blocking=True)
						micro_tokenized_labels = tokenized_labels[micro_idx:micro_end].to(device, non_blocking=True)
						micro_batches.append((micro_images, micro_tokenized_labels))
				# Process each micro-batch
				for micro_idx, (micro_images, micro_tokenized_labels) in enumerate(micro_batches):
						with torch.amp.autocast(device_type=device.type, enabled=True):
								logits_per_image, logits_per_text = model(micro_images, micro_tokenized_labels)
								ground_truth = torch.arange(len(micro_images), dtype=torch.long, device=device)
								loss_img = criterion(logits_per_image, ground_truth)
								loss_txt = criterion(logits_per_text, ground_truth)
								total_loss = 0.5 * (loss_img + loss_txt)
						# Scale the loss to account for accumulation (normalize by accumulation_steps)
						scaled_loss = total_loss / accumulation_steps
						scaler.scale(scaled_loss).backward()
						# Accumulate gradients
						if (micro_idx + 1) % accumulation_steps == 0 or micro_idx == len(micro_batches) - 1:
								torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
								scaler.step(optimizer)
								scaler.update()
								optimizer.zero_grad(set_to_none=True)  # Clear gradients after optimization step
								scheduler.step()
						epoch_loss += total_loss.item() * micro_images.size(0)  # Accumulate loss for reporting
				if bidx % print_every == 0 or bidx + 1 == len(train_loader):
						print(f"Batch [{bidx+1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
		
		avg_training_loss = epoch_loss / len(train_loader.dataset)  # Normalize by total samples
		training_losses.append(avg_training_loss)
		
		# Evaluate on validation set
		metrics_per_epoch = evaluate_loss_and_accuracy(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				device=device,
				topK_values=top_k_values,
		)
		metrics_for_all_epochs.append(metrics_per_epoch)
		
		# Compute retrieval metrics
		img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
				model=model,
				validation_loader=validation_loader,
				device=device,
				topK_values=top_k_values,
		)
		img2txt_metrics_list.append(img2txt_metrics)
		txt2img_metrics_list.append(txt2img_metrics)
		print(
				f'@ Epoch {epoch + 1}:\n'
				f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
				f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
				f'[image retrieval per text]: {metrics_per_epoch.get("txt2img_acc")}'
		)
		# Checkpointing
		current_val_loss = metrics_per_epoch["val_loss"]
		checkpoint = {
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"best_val_loss": best_val_loss,
		}
		if current_val_loss < best_val_loss - early_stopping.min_delta:
				print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
				best_val_loss = current_val_loss
				checkpoint.update({"best_val_loss": best_val_loss})
				torch.save(checkpoint, mdl_fpth)
				best_img2txt_metrics = img2txt_metrics
				best_txt2img_metrics = txt2img_metrics
		# Early stopping
		if early_stopping.should_stop(current_val_loss, model, epoch):
				if current_phase >= min_phases_before_stopping:
						print(f"Early stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
						break
				else:
						print(f"Early stopping condition met at epoch {epoch + 1}! but delaying until minimum phases ({min_phases_before_stopping}) are reached. Current phase: {current_phase}")
		print("-" * 140)
print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))
file_base_name = (
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"ep_{len(training_losses)}_init_lr_{initial_learning_rate:.1e}_final_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_dropout_{dropout_val}"
)
plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"val_acc": os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png"),
		"img2txt_topk": os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png"),
		"txt2img_topk": os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
}
plot_loss_accuracy(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
		val_acc_img2txt_list=[metrics["img2txt_acc"] for metrics in metrics_for_all_epochs],
		val_acc_txt2img_list=[metrics["txt2img_acc"] for metrics in metrics_for_all_epochs],
		img2txt_topk_accuracy_list=[metrics["img2txt_topk_acc"] for metrics in metrics_for_all_epochs],
		txt2img_topk_accuracy_list=[metrics["txt2img_topk_acc"] for metrics in metrics_for_all_epochs],
		mean_reciprocal_rank_list=[metrics["mean_reciprocal_rank"] for metrics in metrics_for_all_epochs],
		cosine_similarity_list=[metrics["cosine_similarity"] for metrics in metrics_for_all_epochs],
		losses_file_path=plot_paths["losses"],
		accuracy_file_path=plot_paths["val_acc"],
		img2txt_topk_accuracy_file_path=plot_paths["img2txt_topk"],
		txt2img_topk_accuracy_file_path=plot_paths["txt2img_topk"],
		mean_reciprocal_rank_file_path=plot_paths["mrr"],
		cosine_similarity_file_path=plot_paths["cs"],
)
plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=plot_paths["retrieval_per_epoch"],
)
plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
)

def handle_phase_transition(
	current_phase: int,
	initial_lr: float,
	max_phases: int,
	scheduler,
):

if current_phase >= max_phases - 1:
	# return current_phase, initial_lr * (0.5 ** current_phase)  # Consistent 2x reduction
	return current_phase, initial_lr * 0.1  # Final phase uses 10% of initial LR

new_phase = current_phase + 1
new_lr = initial_lr * (0.8 ** new_phase)  # Gentle 20% reduction per phase

# update schuler max_lr: Preserve scheduler momentum when possible
if isinstance(scheduler, lr_scheduler.OneCycleLR):
	scheduler.max_lr = new_lr
	for param_group in scheduler.optimizer.param_groups:
		param_group['lr'] = new_lr
else:
	scheduler = lr_scheduler.CosineAnnealingLR(
		scheduler.optimizer, 
		T_max=10, 
		eta_min=new_lr*0.1
	)

print(f"\tTransitioning to Phase {new_phase} with new learning rate {new_lr:.1e}")
return new_phase, new_lr



def get_status(
	model: torch.nn.Module,
	phase: int = 0,
	layers_to_unfreeze: list = [],
	cache: dict = None,
	print_report: bool = True,
) -> dict:
# Get layer information with caching
layer_info = count_clip_layers(model, cache=cache)
total_layers = layer_info['counts']['total']

# Layer freezing statistics
frozen_layer_set = set(layers_to_unfreeze)
total_frozen = len(frozen_layer_set & layer_info['all_layers'])
frozen_layer_percent = round((total_frozen / total_layers) * 100, 2) if total_layers > 0 else 0.0
# Prepare statistics
stats = {
		'parameters': {
				'total': total_params,
				'trainable': trainable_params,
				'frozen': frozen_params,
				'trainable_percent': trainable_percent,
				'frozen_percent': frozen_percent,
		},
		'layers': {
				'total': total_layers,
				'frozen': total_frozen,
				'frozen_percent': frozen_layer_percent,
				'categories': {
						k: {
								'total': v,
								'frozen': len(frozen_layer_set & layer_info['categories'][k]),
								'frozen_percent': round(
										(len(frozen_layer_set & layer_info['categories'][k]) / v) * 100, 2
								) if v > 0 else 0.0
						}
						for k, v in layer_info['counts'].items() if k != 'total'
				}
		}
}
# Print formatted report if requested
if print_report:
		print("\n" + "="*80)
		print(f"{model.__class__.__name__} {model.name} Status - Phase {phase}".center(80))
		print("="*80)
		# Parameter table
		param_table = [
				["Total Parameters", f"{stats['parameters']['total']:,}"],
				["Trainable Parameters", f"{stats['parameters']['trainable']:,} ({stats['parameters']['trainable_percent']:.2f}%)"],
				["Frozen Parameters", f"{stats['parameters']['frozen']:,} ({stats['parameters']['frozen_percent']:.2f}%)"]
		]
		print("\nParameter Statistics:")
		print(tabulate.tabulate(param_table, tablefmt="grid"))
		# Layer table
		layer_table = [
				["Total Layers", stats['layers']['total']],
				["Frozen Layers", f"{stats['layers']['frozen']} ({stats['layers']['frozen_percent']:.2f}%)"]
		]
		print("\nLayer Statistics:")
		print(tabulate.tabulate(layer_table, tablefmt="grid"))
		# Category breakdown (dynamic table generation)
		category_table = [
				[
						cat.replace('_', ' ').title(),
						f"{data['frozen']}/{data['total']}",
						f"{data['frozen_percent']:.2f}%"
				]
				for cat, data in stats['layers']['categories'].items()
		]
		print("\nLayer Category Breakdown:")
		print(tabulate.tabulate(category_table, headers=["Category", "Frozen/Total", "% Frozen"], tablefmt="grid"))
		print("\n" + "="*80 + "\n")

return stats


# def get_freeze_schedule(model: torch.nn.Module):
# 	layer_groups = get_layer_groups(model=model)

# 	total_v_layers = len(layer_groups['visual_transformer'])
# 	total_t_layers = len(layer_groups['text_transformer'])
# 	print(f"Total visual layers: {total_v_layers} | 80%: {int(0.8*total_v_layers)} 60%: {int(0.6*total_v_layers)} 40%: {int(0.4*total_v_layers)}")
# 	print(f"Total text layers: {total_t_layers} | 80%: {int(0.8*total_t_layers)} 60%: {int(0.6*total_t_layers)} 40%: {int(0.4*total_t_layers)}")

# 	schedule = [
# 		# Phase 0: Freeze all layers except the projection layers:
# 		layer_groups['visual_frontend'] + layer_groups['visual_transformer'] + layer_groups['text_frontend'] + layer_groups['text_transformer'],
# 		# Phase 1: Freeze 80% of transformer blocks:
# 		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.8*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.8*total_t_layers)],
# 		# Phase 2: freeze 60% of transformer blocks:
# 		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.6*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.6*total_t_layers)],
# 		# Phase 3: freeze 40% of transformer blocks:
# 		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.4*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.4*total_t_layers)],
# 		# Phase 4: freeze only (visual + text) frontends
# 		layer_groups['visual_frontend'] + layer_groups['text_frontend']
# 	]
# 	return schedule


def progressive_freeze_finetune(
	model:torch.nn.Module,
	train_loader:DataLoader,
	validation_loader:DataLoader,
	num_epochs:int,
	nw:int,
	print_every:int,
	learning_rate:float,
	weight_decay:float,
	device:str,
	results_dir:str,
	window_size:int=10,
	patience:int=10,
	min_delta:float=1e-4,
	cumulative_delta:float=5e-3,
	minimum_epochs:int=20,
	TOP_K_VALUES:List[int]=[1, 5, 10, 15, 20],
):

early_stopping = EarlyStopping(
	patience=patience,									# Wait for 10 epochs without improvement before stopping
	min_delta=min_delta,								# Consider an improvement only if the change is greater than 0.0001
	cumulative_delta=cumulative_delta,	# Cumulative improvement over the window should be greater than 0.005
	window_size=window_size,						# Consider the last 10 epochs for cumulative trend
	mode='min',													# Minimize loss
	min_epochs=minimum_epochs,					# Ensure at least 20 epochs of training
	restore_best_weights=True						# Restore model weights to the best epoch
)

try:
	dataset_name = validation_loader.dataset.dataset.__class__.__name__ # CIFAR10, ImageNet, etc.
except:
	dataset_name = validation_loader.dataset.dataset_name # 

os.makedirs(results_dir, exist_ok=True)
mode = inspect.stack()[0].function
model_arch = model.name
model_name = model.__class__.__name__
print(f"{mode} {model_name} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))
if torch.cuda.is_available():
	print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

dropout_val = None
for name, module in model.named_modules():
	# print(f"{name}: {type(module).__name__}")
	if isinstance(module, torch.nn.Dropout):
		# print(f"{name}.p: {module.p}")
		dropout_val = module.p
		break
if dropout_val is None:
	dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found (unlikely in your case)

layer_cache = {}  # Initialize cache for layer counts
freeze_schedule = get_freeze_schedule(
	model=model,
	freeze_percentages=[1.0, 0.8, 0.6, 0.4, 0.0],  # Default percentages
	layer_groups_to_freeze=['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer']
)
print(json.dumps(freeze_schedule, indent=2, ensure_ascii=False))

mdl_fpth = os.path.join(
	results_dir,
	f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
	f"dropout_{dropout_val}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
)

criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler(
	device=device,
	init_scale=2**16,
	growth_factor=2.0,
	backoff_factor=0.5,
	growth_interval=2000,
)
training_losses = []
img2txt_metrics_list = []
txt2img_metrics_list = []
metrics_for_all_epochs = []
train_start_time = time.time()
best_val_loss = float('inf')
best_img2txt_metrics = None
best_txt2img_metrics = None
current_phase = 0
plateau_threshold = min_delta # ensure parameter consistency
initial_learning_rate = learning_rate # Store the initial value

for epoch in range(num_epochs):
	torch.cuda.empty_cache()  # Free up GPU memory
	print(f"Epoch [{epoch+1}/{num_epochs}]")
	# Adaptive Progressive Layer Freezing Schedule:
	if epoch > 1: # 2 epochs at least needed to compare
		should_transition = should_transition_phase(
			losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
			th=plateau_threshold,
			window=window_size,
		)
		if should_transition:
			print(f"Plateau detected @ Epoch: {epoch+1} Transitioning from phase: {current_phase} to next phase.")
			current_phase, learning_rate = handle_phase_transition(
				current_phase=current_phase,
				initial_lr=initial_learning_rate,
				max_phases=len(freeze_schedule)
			)
	# Freeze layers based on the current phase
	freeze_layers(
		model=model, 
		strategy=freeze_schedule,
		phase=current_phase,
	)
	optimizer = AdamW(
		params=filter(lambda p: p.requires_grad, model.parameters()),
		lr=learning_rate, # potentially update learning rate based on phase
		betas=(0.9, 0.98),
		eps=1e-8,
		weight_decay=weight_decay,
	)
	scheduler = lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs - epoch, # Adjust for remaining epochs
		pct_start=0.1,
		anneal_strategy='cos',
	)
	epoch_loss = 0.0
	for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
		optimizer.zero_grad() # Clear gradients from previous batch
		images = images.to(device, non_blocking=True)
		tokenized_labels = tokenized_labels.to(device, non_blocking=True)

		with torch.amp.autocast(device_type=device.type, enabled=True): # # Automatic Mixed Precision (AMP) backpropagation:
			logits_per_image, logits_per_text = model(images, tokenized_labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
			ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
			loss_img = criterion(logits_per_image, ground_truth)
			loss_txt = criterion(logits_per_text, ground_truth)
			total_loss = 0.5 * (loss_img + loss_txt)
		scaler.scale(total_loss).backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # stabilize training if exploding gradients
		scaler.step(optimizer)
		scaler.update()
		scheduler.step() # Update learning rate
		if bidx%print_every==0 or bidx+1==len(train_loader):
			print(
				f"\t\tBatch [{bidx+1}/{len(train_loader)}] "
				f"Loss: {total_loss.item():.7f}",
			)
		epoch_loss += total_loss.item()

	avg_training_loss = epoch_loss / len(train_loader)
	training_losses.append(avg_training_loss)

	metrics_per_epoch = evaluate_loss_and_accuracy(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		device=device,
		topK_values=TOP_K_VALUES,
	)
	metrics_for_all_epochs.append(metrics_per_epoch)
	print(
		f'@ Epoch {epoch + 1}:\n'
		f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
		f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
		f'[image retrieval per text]: {metrics_per_epoch.get("txt2img_acc")}'
	)

	# Compute retrieval-based metrics
	img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
		model=model,
		validation_loader=validation_loader,
		device=device,
		topK_values=TOP_K_VALUES,
	)
	img2txt_metrics_list.append(img2txt_metrics)
	txt2img_metrics_list.append(txt2img_metrics)

	# Early stopping
	current_val_loss = metrics_per_epoch["val_loss"]
	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"best_val_loss": best_val_loss,
	}

	if current_val_loss < best_val_loss - early_stopping.min_delta:
		print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
		best_val_loss = current_val_loss
		checkpoint.update({"best_val_loss": best_val_loss})
		torch.save(checkpoint, mdl_fpth)
		best_img2txt_metrics = img2txt_metrics
		best_txt2img_metrics = txt2img_metrics

	if early_stopping.should_stop(current_val_loss, model, epoch):
		print(f"\nEarly stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
		final_metrics = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=TOP_K_VALUES,
		)

		final_img2txt, final_txt2img = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=TOP_K_VALUES,
		)

		metrics_per_epoch = final_metrics
		img2txt_metrics = final_img2txt
		txt2img_metrics = final_txt2img

		if final_metrics["val_loss"] < best_val_loss:
			best_val_loss = final_metrics["val_loss"]
			checkpoint.update({"best_val_loss": best_val_loss})
			best_img2txt_metrics = final_img2txt
			best_txt2img_metrics = final_txt2img
			torch.save(checkpoint, mdl_fpth)
		break
	print("-" * 170)
print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(150, "-"))

file_base_name = (
	f"{dataset_name}_{mode}_{re.sub('/', '', model_arch)}_"
	f"ep_{len(training_losses)}_lr_{learning_rate:.1e}_"
	f"wd_{weight_decay:.1e}_bs_{train_loader.batch_size}_do_{dropout_val}"
)

losses_fpth = os.path.join(results_dir, f"{file_base_name}_losses.png")
val_acc_fpth = os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png")
img2txt_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png")
txt2img_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png")
mrr_fpth = os.path.join(results_dir, f"{file_base_name}_mrr.png")
cs_fpth = os.path.join(results_dir, f"{file_base_name}_cos_sim.png")	

plot_loss_accuracy(
	dataset_name=dataset_name,
	train_losses=training_losses,
	val_losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
	val_acc_img2txt_list=[metrics["img2txt_acc"] for metrics in metrics_for_all_epochs],
	val_acc_txt2img_list=[metrics["txt2img_acc"] for metrics in metrics_for_all_epochs],
	img2txt_topk_accuracy_list=[metrics["img2txt_topk_acc"] for metrics in metrics_for_all_epochs],
	txt2img_topk_accuracy_list=[metrics["txt2img_topk_acc"] for metrics in metrics_for_all_epochs],
	mean_reciprocal_rank_list=[metrics["mean_reciprocal_rank"] for metrics in metrics_for_all_epochs],
	cosine_similarity_list=[metrics["cosine_similarity"] for metrics in metrics_for_all_epochs],
	losses_file_path=losses_fpth,
	accuracy_file_path=val_acc_fpth,
	img2txt_topk_accuracy_file_path=img2txt_topk_accuracy_fpth,
	txt2img_topk_accuracy_file_path=txt2img_topk_accuracy_fpth,
	mean_reciprocal_rank_file_path=mrr_fpth,
	cosine_similarity_file_path=cs_fpth,
)

retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
plot_retrieval_metrics_per_epoch(
	dataset_name=dataset_name,
	image_to_text_metrics_list=img2txt_metrics_list,
	text_to_image_metrics_list=txt2img_metrics_list,
	fname=retrieval_metrics_fpth,
)

retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
plot_retrieval_metrics_best_model(
	dataset_name=dataset_name,
	image_to_text_metrics=best_img2txt_metrics,
	text_to_image_metrics=best_txt2img_metrics,
	fname=retrieval_metrics_best_model_fpth,
)


def count_clip_layers(model: torch.nn.Module) -> dict:
	"""
	Accurately counts and categorizes CLIP model layers for all architectures
	Returns layer categories and total count
	"""
	layer_categories = defaultdict(set)
	for name, _ in model.named_parameters():
			parts = name.split('.')
			parent_layer = '.'.join(parts[:-1])  # Get parent module
			
			# Visual components
			if 'visual' in parts:
					if 'transformer' in parts:
							# ViT models: group by transformer block
							idx = parts.index('transformer') + 2
							base = '.'.join(parts[:idx])
							layer_categories['Visual Transformer'].add(base)
					elif 'layer' in parts:
							# ResNet models: group by CNN layer block
							idx = parts.index('layer') + 2
							base = '.'.join(parts[:idx])
							layer_categories['Visual CNN'].add(base)
					elif any(x in parts for x in ['conv1', 'class_embedding', 'positional_embedding']):
							layer_categories['Visual Frontend'].add(parent_layer)
					elif any(x in parts for x in ['proj', 'ln_post']):
							layer_categories['Visual Projection'].add(parent_layer)
					else:
							layer_categories['Visual Other'].add(parent_layer)
			
			# Text components
			elif 'text' in parts or 'token_embedding' in name:
					if 'transformer' in parts:
							idx = parts.index('transformer') + 2
							base = '.'.join(parts[:idx])
							layer_categories['Text Transformer'].add(base)
					elif any(x in parts for x in ['token_embedding', 'positional_embedding']):
							layer_categories['Text Frontend'].add(parent_layer)
					elif 'text_projection' in parts:
							layer_categories['Text Projection'].add(parent_layer)
					else:
							layer_categories['Text Other'].add(parent_layer)
			
			# Other components
			else:
					layer_categories['Other'].add(parent_layer)
	# Convert to counts and add metadata
	counts = {k: len(v) for k, v in layer_categories.items()}
	counts['total'] = sum(counts.values())
	return {
			'categories': layer_categories,
			'counts': counts,
			'all_layers': {layer for v in layer_categories.values() for layer in v}
	}

def get_status(
		model: torch.nn.Module,
		phase: int = 0,
		layers_to_freeze: list = [],
	):
	"""
	Comprehensive model status reporting with architecture-aware statistics
	"""
	# Get layer information
	layer_info = count_clip_layers(model)
	total_layers = layer_info['counts']['total']
	
	# Parameter statistics
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	total_params = sum(p.numel() for p in model.parameters()) # trainable_params + frozen_params
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100

	# Layer freezing statistics
	frozen_layer_set = set(layers_to_freeze)
	total_frozen = len(frozen_layer_set & layer_info['all_layers'])
	
	# Prepare statistics
	stats = {
			'parameters': {
					'total': total_params,
					'trainable': trainable_params,
					'frozen': frozen_params,
					'trainable_percent': trainable_percent,
					'frozen_percent': frozen_percent,
			},
			'layers': {
					'total': total_layers,
					'frozen': total_frozen,
					'frozen_percent': (total_frozen / total_layers) * 100 if total_layers > 0 else 0,
					'categories': {
							k: {
									'total': v,
									'frozen': len(frozen_layer_set & layer_info['categories'][k])
							} 
							for k, v in layer_info['counts'].items() if k != 'total'
					}
			}
	}
	# Print formatted report
	print("\n" + "="*80)
	print(f"{model.__class__.__name__} {model.name} Status - Phase {phase}".center(80))
	print("="*80)
	
	# Parameter table
	param_table = [
		["Total Parameters", f"{stats['parameters']['total']:,}"],
		["Trainable Parameters", f"{stats['parameters']['trainable']:,} ({stats['parameters']['trainable_percent']:.2f}%)"],
		["Frozen Parameters", f"{stats['parameters']['frozen']:,} ({stats['parameters']['frozen_percent']:.2f}%)"]
	]
	print("\nParameter Statistics:")
	print(tabulate.tabulate(param_table, tablefmt="grid"))
	
	# Layer table
	layer_table = [
			["Total Layers", stats['layers']['total']],
			["Frozen Layers", f"{stats['layers']['frozen']} ({stats['layers']['frozen_percent']:.1f}%)"]
	]
	print("\nLayer Statistics:")
	print(tabulate.tabulate(layer_table, tablefmt="grid"))
	
	# Category breakdown
	category_table = []
	for cat, data in stats['layers']['categories'].items():
		category_table.append([
			cat.replace('_', ' ').title(),
			f"{data['frozen']}/{data['total']}",
			f"{(data['frozen']/data['total'])*100:.1f}%" if data['total'] > 0 else "0.0%"
		])
	
	print("\nLayer Category Breakdown:")
	print(tabulate.tabulate(category_table, headers=["Category", "Frozen/Total", "% Frozen"], tablefmt="grid"))
	print("\n" + "="*80 + "\n")
	return stats

# def finetune(
# 		model:torch.nn.Module,
# 		train_loader:DataLoader,
# 		validation_loader:DataLoader,
# 		num_epochs:int,
# 		nw:int,
# 		print_every:int,
# 		learning_rate:float,
# 		weight_decay:float,
# 		device:str,
# 		results_dir:str,
# 		window_size:int=10,
# 		patience:int=10,
# 		min_delta:float=1e-4,
# 		cumulative_delta:float=5e-3,
# 		minimum_epochs:int=20,
# 		TOP_K_VALUES:List[int]=[1, 5, 10, 15, 20],
# 	):
# 	early_stopping = EarlyStopping(
# 		patience=patience,									# Wait for 10 epochs without improvement before stopping
# 		min_delta=min_delta,								# Consider an improvement only if the change is greater than 0.0001
# 		cumulative_delta=cumulative_delta,	# Cumulative improvement over the window should be greater than 0.005
# 		window_size=window_size,						# Consider the last 10 epochs for cumulative trend
# 		mode='min',													# Minimize loss
# 		min_epochs=minimum_epochs,					# Ensure at least 20 epochs of training
# 		restore_best_weights=True						# Restore model weights to the best epoch
# 	)
# 	try:
# 		dataset_name = validation_loader.dataset.dataset.__class__.__name__ # CIFAR10, ImageNet, etc.
# 	except AttributeError as e:
# 		dataset_name = validation_loader.dataset.dataset_name # 
# 	os.makedirs(results_dir, exist_ok=True)
# 	mode = finetune.__name__
# 	model_arch = model.name
# 	model_name = model.__class__.__name__
# 	print(f"{mode} {model_name} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) | {type(device)} {device} [x{nw} cores]".center(160, "-"))
# 	if torch.cuda.is_available():
# 		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

# 	dropout_val = None
# 	for name, module in model.named_modules():
# 		# print(f"{name}: {type(module).__name__}")
# 		if isinstance(module, torch.nn.Dropout):
# 			# print(f"{name}.p: {module.p}")
# 			dropout_val = module.p
# 			break
# 	if dropout_val is None:
# 		dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found (unlikely in your case)


# def evaluate_retrieval_performance(
# 	model: torch.nn.Module,
# 	validation_loader: DataLoader,
# 	device: str="cuda:0",
# 	topK_values: List=[1, 3, 5],
# 	):
# 	dataset_name = validation_loader.name
# 	model_name = model.__class__.__name__
# 	model_arch = model.name
# 	print(f">> Evaluating {model_name} - {model_arch} Retrieval Performance [{dataset_name}]: {topK_values}...")
# 	model.eval() # dropout is disabled, ensuring deterministic outputs

# 	image_embeddings = []
# 	image_labels = []

# 	try:
# 		class_names = validation_loader.dataset.dataset.classes
# 	except:
# 		class_names = validation_loader.dataset.unique_labels
# 	n_classes = len(class_names)
# 	# print(f"{n_classes} Class [Validation]:\n{class_names}")
# 	with torch.no_grad():
# 		text_inputs = clip.tokenize(texts=class_names).to(device, non_blocking=True)
# 		class_text_embeddings = model.encode_text(text_inputs)
# 		class_text_embeddings = class_text_embeddings / class_text_embeddings.norm(dim=-1, keepdim=True)
		
# 		for bidx, (images, _, class_indices) in enumerate(validation_loader):
# 			images = images.to(device, non_blocking=True)
# 			class_indices = class_indices.to(device, non_blocking=True)
# 			# print("Sample class indices:", class_indices[:10].cpu().numpy())
# 			# print("Corresponding class names:", [class_names[i] for i in class_indices[:10].cpu().numpy()])
# 			image_embeds = model.encode_image(images)
# 			image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
			
# 			image_embeddings.append(image_embeds.cpu().numpy())
# 			image_labels.extend(class_indices.cpu().numpy())

# 	# Aggregate and normalize embeddings
# 	image_embeddings = np.concatenate(image_embeddings, axis=0)
# 	image_labels = np.array(image_labels)
# 	class_text_embeddings = class_text_embeddings.cpu().numpy()
	
# 	# Compute similarity matrix
# 	similarity_matrix = image_embeddings @ class_text_embeddings.T
# 	# logit_scale = model.logit_scale.exp().detach().cpu().numpy()  # Detach before converting to NumPy
# 	# print(logit_scale, type(logit_scale), logit_scale.shape, logit_scale.dtype)
# 	# # similarity_matrix = logit_scale * (image_embeddings @ class_text_embeddings.T)
# 	print("Similarity matrix stats:")
# 	print(
# 		type(similarity_matrix),
# 		similarity_matrix.shape,
# 		similarity_matrix.dtype,
# 		similarity_matrix.min(),
# 		similarity_matrix.max(),
# 		similarity_matrix.mean(),
# 		similarity_matrix.std(),
# 	)
# 	print(similarity_matrix[:10, :10]) # ensure values are reasonable (e.g., -1 to 1).

# 	image_to_text_metrics = get_retrieval_metrics(
# 		similarity_matrix=similarity_matrix,
# 		query_labels=image_labels,
# 		candidate_labels=np.arange(n_classes),
# 		topK_values=topK_values,
# 		mode="Image-to-Text",
# 		class_counts=None,  # No class counts for Image-to-Text
# 		max_k=n_classes,  # Pass max_k for Image-to-Text to limit K to the number of classes
# 	)
	
# 	text_to_image_metrics = get_retrieval_metrics(
# 		similarity_matrix=class_text_embeddings @ image_embeddings.T,
# 		query_labels=np.arange(n_classes),
# 		candidate_labels=image_labels,
# 		topK_values=topK_values,
# 		mode="Text-to-Image",
# 		class_counts=np.bincount(image_labels), # Count number of occurrences of each value in array of non-negative ints.
# 		max_k=None,  # No limit on K for Text-to-Image
# 	)

# 	return image_to_text_metrics, text_to_image_metrics

# def get_retrieval_metrics(
# 	similarity_matrix: np.ndarray,
# 	query_labels: np.ndarray,
# 	candidate_labels: np.ndarray,
# 	topK_values: List[int] = [1, 3, 5],
# 	mode: str ="Image-to-Text",
# 	class_counts: np.ndarray = None,
# 	max_k: int = None,  # New parameter to limit K values (None for no limit)
# 	):
# 	num_queries, num_candidates = similarity_matrix.shape
# 	assert num_queries == len(query_labels), "Number of queries must match labels"
	
# 	num_classes = len(np.unique(candidate_labels)) # unique values in candidate_labels
# 	if max_k is not None:
# 		valid_K_values = [K for K in topK_values if K <= max_k]
# 	else:
# 		valid_K_values = topK_values # No limit on K values

# 	if len(valid_K_values) < len(topK_values):
# 		print(f"<!> Warning: K values: ({set(topK_values) - set(valid_K_values)}) exceed the number of classes ({num_classes}). => ignored!")
	
# 	metrics = {
# 		"mP": {},
# 		"mAP": {},
# 		"Recall": {},
# 	}
	
# 	for K in valid_K_values:
# 		top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :K]
		
# 		precision, recall, ap = [], [], []
# 		for i in range(num_queries):
# 			true_label = query_labels[i]
# 			retrieved_labels = candidate_labels[top_k_indices[i]]
# 			correct = np.sum(retrieved_labels == true_label)
			
# 			# 1. Precision @ K
# 			precision.append(correct / K)
			
# 			# 2. Compute Recall@K with division by zero protection
# 			if mode == "Image-to-Text":
# 				relevant_count = 1  # Single relevant item per query [single label per image]
# 			else:
# 				relevant_count = class_counts[true_label] if class_counts is not None else 0
					
# 			if relevant_count == 0:
# 				recall.append(0.0)
# 			else:
# 				recall.append(correct / relevant_count)

# 			# 3. Compute AP@K with proper normalization
# 			relevant_positions = np.where(retrieved_labels == true_label)[0]
# 			p_at = []
# 			cumulative_correct = 0

# 			for pos in relevant_positions:
# 				if pos < K:  # Only consider positions within top-K
# 					cumulative_correct += 1
# 					precision_at_rank = cumulative_correct / (pos + 1)  # pos is 0-based
# 					p_at.append(precision_at_rank)

# 			# Determine normalization factor
# 			if mode == "Image-to-Text":
# 				R = 1  # Always 1 relevant item for image-to-text
# 			else:
# 				R = class_counts[true_label] if class_counts is not None else 0
					
# 			# Handle queries with no relevant items
# 			if R == 0:
# 				ap.append(0.0)
# 				continue
					
# 			if len(p_at) == 0:
# 				ap.append(0.0)
# 			else:
# 				ap.append(sum(p_at) / min(R, K)) # Normalize by min(R, K)

# 		# Store metrics for this K
# 		metrics["mP"][str(K)] = np.mean(precision)
# 		metrics["mAP"][str(K)] = np.mean(ap)
# 		metrics["Recall"][str(K)] = np.mean(recall)
	
# 	return metrics


print(f"PyTorch seed: {torch.initial_seed()} NumPy seed: {np.random.get_state()[1][0]}")
if torch.cuda.is_available():
	print("PyTorch CUDA seed: ", torch.cuda.initial_seed())
print(f"PyTorch random number: {torch.randn(1)}) NumPy random number: {np.random.rand(1)}")


class IMAGE_TEXT_DATASET(Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
		self.tokenized_labels = clip.tokenize(texts=[dataset.classes[lbl_idx] for i, (img, lbl_idx) in enumerate(self.dataset)])

	def __getitem__(self, index):
		img = self.dataset[index][0]
		tokenized_lbl = self.tokenized_labels[index]
		return img, tokenized_lbl

	def __len__(self):
		return len(self.dataset)

	def __repr__(self):
		return f"IMAGE_TEXT_DATASET\n{self.dataset}\nlabels={self.tokenized_labels.shape}"


def get_text_to_image_linear_probe_accuracy(
	train_dataset,
	val_dataset,
	model,
	preprocess,
	device: str = "cuda:0",
	batch_size: int = 64,
	seed: int = 42
	):
	"""
	Compute Linear Probe Accuracy for Text-to-Image retrieval.
	:param train_dataset: Training dataset.
	:param val_dataset: Validation dataset.
	:param model: CLIP model.
	:param preprocess: Preprocessing for images.
	:param device: Device to run computations on.
	:param batch_size: Batch size for processing images.
	:param seed: Random seed for reproducibility.
	:return: Linear Probe Accuracy.
	"""
	print(f"Text-to-Image Linear Probe Accuracy".center(160, " "))
	
	# Extract text features from labels
	def get_text_features(dataset):
			labels = sorted(list(set(dataset["label"].tolist())))
			text_inputs = clip.tokenize(labels).to(device)
			with torch.no_grad():
					text_features = model.encode_text(text_inputs)
			text_features /= text_features.norm(dim=-1, keepdim=True)
			return text_features.cpu().numpy(), labels
	
	train_features, train_labels = get_text_features(train_dataset)
	val_features, val_labels = get_text_features(val_dataset)
	print(f"Training features[{type(train_features)}]: {train_features.shape}")
	print(f"Validation features[{type(val_features)}]: {val_features.shape}")
	# Label mappings
	label_dict = {lbl: idx for idx, lbl in enumerate(train_labels)}
	train_labels_int = [label_dict[lbl] for lbl in train_dataset["label"].tolist()]
	val_labels_int = [label_dict[lbl] for lbl in val_dataset["label"].tolist()]
	print(f"Training labels[{type(train_labels_int)}]: {len(train_labels_int)}")
	print(f"Validation labels[{type(val_labels_int)}]: {len(val_labels_int)}")
	# Train logistic regression
	classifier = LogisticRegression(
			random_state=seed,
			C=0.316,
			max_iter=1000,
			tol=1e-4,
			verbose=1,
			solver='saga',
			n_jobs=-1
	)
	classifier.fit(train_features, train_labels_int)
	
	# Evaluate
	val_features = clip.tokenize(val_labels).to(device)
	with torch.no_grad():
			val_features = model.encode_text(val_features).cpu().numpy()
	predictions = classifier.predict(val_features)
	
	return np.mean(predictions == val_labels_int)

def get_text_to_image_zero_shot_accuracy(
		dataset,
		model,
		preprocess,
		K: int = 5,
		device: str = "cuda:0",
		batch_size: int = 64,
		image_features_file: str = "validation_image_features.gz"
	):
		"""
		Compute Zero Shot Accuracy for Text-to-Image retrieval.
		:param dataset: Validation dataset with image paths and labels.
		:param model: CLIP model.
		:param preprocess: Preprocessing for images.
		:param K: Number of top predictions to consider.
		:param device: Device to run computations on.
		:param batch_size: Batch size for processing images.
		:param image_features_file: Path to precomputed image features.
		:return: Zero Shot Accuracy.
		"""
		print(f"Text-to-Image Zero Shot Accuracy (K={K})".center(160, " "))
		# Create label-to-integer mapping
		label_dict = {label: label_int for label, label_int in zip(dataset["label"], dataset["label_int"])}
		# Load or compute image features
		if not os.path.exists(image_features_file):
				image_features = []
				for i in range(0, len(dataset["img_path"]), batch_size):
						batch_paths = dataset["img_path"][i:i+batch_size]
						batch_tensors = torch.stack([preprocess(Image.open(path)).to(device) for path in batch_paths])
						with torch.no_grad():
								batch_features = model.encode_image(batch_tensors)
								batch_features /= batch_features.norm(dim=-1, keepdim=True)
						image_features.extend(batch_features.cpu().numpy())
				image_features = np.array(image_features)
				np.save(image_features_file, image_features)
		else:
				image_features = np.load(image_features_file)
		
		# Get unique labels to use as text queries
		labels = sorted(list(set(dataset["label"].tolist())))
		text_inputs = clip.tokenize(labels).to(device)
		
		# Compute text features for these labels
		with torch.no_grad():
				text_features = model.encode_text(text_inputs)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		text_features = text_features.cpu().numpy()
		
		# Normalize image features
		image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
		
		# Compute similarities and retrieve top-K images
		similarities = text_features @ image_features.T  # (num_labels, num_images)
		top_k_indices = np.argsort(-similarities, axis=-1)[:, :K]
		
		# Calculate accuracy: Check if any of the top-K images match the ground-truth label
		ground_truth = np.array(dataset["label_int"].tolist())
		accuracies = []
		for label_idx, label in enumerate(labels):
				true_indices = np.where(ground_truth == label_dict[label])[0]
				retrieved_indices = top_k_indices[label_idx]
				count = len(set(retrieved_indices) & set(true_indices))
				accuracies.append(count > 0)
		zero_shot_accuracy = np.mean(accuracies)
		print(f"Top-{K} Zero-Shot Accuracy: {zero_shot_accuracy:.3f}")
		return zero_shot_accuracy


prec_at_k = []
recall_at_k = []
for i, label_features in enumerate(tokenized_labels_features):
	sim = (100.0 * label_features @ all_image_features.T).softmax(dim=-1) # similarities between query and all images
	topk_probs, topk_indices = sim.topk(K, dim=-1)
	topk_pred_labels_idxs = [dataset_labels_int[topk_indices.squeeze().item()]] if K==1 else [dataset_labels_int[idx] for idx in topk_indices.squeeze().cpu().numpy()] # K@1, 5, ...
	relevant_retrieved_images_for_label_i = topk_pred_labels_idxs.count(i)# count number of relevant images (i.e., images with the same label) in top-K retrieved images.
	prec_at_k.append(relevant_retrieved_images_for_label_i/K)
	all_images_with_label_i = [idx for idx, (img, lbl) in enumerate(zip(dataset_images_id, dataset_labels_int)) if lbl == i]
	num_all_images_with_label_i = len(all_images_with_label_i)
	recall_at_k.append(relevant_retrieved_images_for_label_i/num_all_images_with_label_i)
avg_prec_at_k = sum(prec_at_k)/len(labels)
avg_recall_at_k = sum(recall_at_k) / len(labels)
print(f"Precision@{K}: {avg_prec_at_k:.3f} {np.mean(prec_at_k)}")
print(f"Recall@{K}: {avg_recall_at_k} {np.mean(recall_at_k)}")
print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

############################################################################################################################
# Check for plateau to adapt phases of progressive freezing
# if epoch > 0 and len(validation_losses) > 1:
# 	current_smoothed_loss = smooth_(losses=validation_losses, window=3)
# 	smoothed_val_losses.append(current_smoothed_loss)
# 	if len(smoothed_val_losses) > 1:
# 		loss_diff = smoothed_val_losses[-2] - smoothed_val_losses[-1]
# 		if loss_diff < plateau_threshold:
# 			counter += 1
# 			print(f"Plateau counter: {counter}/{patience_per_phase} (Smoothed loss: {current_smoothed_loss:.6f})")
# 		else:
# 			counter = 0
# 			print(f"No plateau detected. Continuing current phase. (Smoothed loss: {current_smoothed_loss:.6f})")
# 		if counter >= patience_per_phase and current_phase < len(freeze_schedule) - 1:
# 			current_phase += 1
# 			counter = 0
# 			learning_rate = initial_learning_rate * (0.1 ** current_phase) # Reduce learning rate by 10x for each new phase
# 			print(f"Plateau detected. Transitioning to Phase {current_phase} with updated LR: {learning_rate:.1e}")
############################################################################################################################


>> Fine-tuning a pre-trained model using conventional backpropagation:
# logits_per_image: similarity scores between each image embedding and all text embeddings in the batch
# Each row in logits_per_image corresponds to one image in the batch, and each column corresponds to a text description.

# logits_per_text: similarity scores between each text embedding and all image embeddings in the batch

# # Conventional backpropagation:
# logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
# ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
# loss_img = criterion(logits_per_image, ground_truth) 
# loss_txt = criterion(logits_per_text, ground_truth)
# total_loss = 0.5 * (loss_img + loss_txt)
# total_loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# optimizer.step() # Update weights

# def evaluate(model, test_loader, criterion, device:str="cuda"):
# 	model.eval()
# 	total_loss = 0
# 	total_correct_text_description_for_each_image = 0
# 	total_correct_image_for_each_text_description = 0
# 	with torch.no_grad():
# 		for batch_idx, (images, labels) in enumerate(test_loader):
# 			images, labels = images.to(device), labels.to(device)
# 			logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
# 			_, predicted_idxs_imgs = torch.max(input=logits_per_image, dim=1, keepdim=True)
# 			_, predicted_idxs_txts = torch.max(input=logits_per_text, dim=1, keepdim=True)
# 			correct_text_description_idxs = torch.argmax(labels, dim=1) # indices of correct text descriptions for each image
# 			# Compare predicted indexes with the correct indexes
# 			total_correct_text_description_for_each_image += (predicted_idxs_imgs == correct_text_description_idxs.unsqueeze(1)).sum().item()
# 			total_correct_image_for_each_text_description += (predicted_idxs_txts == correct_text_description_idxs.unsqueeze(1)).sum().item()

# 			# validation loss
# 			ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
# 			loss_img = criterion(logits_per_image, ground_truth) 
# 			loss_txt = criterion(logits_per_text, ground_truth)
# 			valid_loss = 0.5 * (loss_img + loss_txt)
# 			total_loss += valid_loss.item()
# 	avg_loss = total_loss / len(test_loader)
# 	accuracy_text_description_for_each_image = total_correct_text_description_for_each_image / len(test_loader.dataset)
# 	accuracy_text_image_for_each_text_description = total_correct_image_for_each_text_description / len(test_loader.dataset)
# 	return avg_loss, accuracy_text_description_for_each_image, accuracy_text_image_for_each_text_description


more advanced:
def evaluate(model, test_loader, criterion, device="cuda"):
	model.eval()
	total_loss = 0
	correct_text_description = 0
	correct_image_for_text = 0
	total_samples = 0
	with torch.no_grad():
		for bidx, (images, labels) in enumerate(test_loader):
			images, labels = images.to(device), labels.to(device)
			batch_size = images.size(0)
			total_samples += batch_size
			logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
			# Predictions
			predicted_text_idxs = torch.argmax(input=logits_per_image, dim=1) # indices of maximum value of all elements in input tensor. torch.Size([batch_size])
			predicted_image_idxs = torch.argmax(input=logits_per_text, dim=1)
			correct_labels = torch.arange(start=0, end=batch_size, dtype=torch.long, device=device) # ground truth labels for each batch item torch.Size([batch_size])
			# Metrics
			correct_text_description += (predicted_text_idxs == correct_labels).sum().item()
			correct_image_for_text += (predicted_image_idxs == correct_labels).sum().item()
			# Validation loss
			loss_img = criterion(logits_per_image, correct_labels)
			loss_txt = criterion(logits_per_text, correct_labels)
			total_loss += 0.5 * (loss_img.item() + loss_txt.item())
	# Compute average loss and accuracies
	avg_loss = total_loss / len(test_loader)
	accuracy_text_description = correct_text_description / total_samples
	accuracy_image_for_text = correct_image_for_text / total_samples
	return avg_loss, accuracy_text_description, accuracy_image_for_text


###################################################################################
# GPU cosine similarity + Average recommendation vector:
def get_customized_cosine_similarity_gpu(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0, batch_size:int=2048):
		print(f"[GPU Optimized] Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]}) batch_size={batch_size}".center(130, "-"))
		print(
			f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype} non_zeros={np.count_nonzero(query_vec)} (ratio={np.count_nonzero(query_vec) / query_vec.size})\n"
			f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
			f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
			f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
		)
		# Clear memory before starting
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()
		
		# Print GPU device information
		device = cp.cuda.Device()
		device_id = device.id
		device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
		print(f"GPU: {device_name} ({device_id})")
		print(f"Initial Free GPU Memory: {device.mem_info[0] / 1024 ** 3:.2f} GB / Total GPU Memory: {device.mem_info[1] / 1024 ** 3:.2f} GB")

		st_t = time.time()
		# Convert inputs to CuPy arrays (float32)
		query_vec_squeezed = cp.asarray(query_vec.ravel(), dtype=cp.float32)
		idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
		spMtx_norm = cp.asarray(spMtx_norm, dtype=cp.float32)

		# Convert sparse matrix to CuPy CSR format
		spMtx_csr = spMtx.tocsr()
		spMtx_gpu = cp.sparse.csr_matrix(
				(cp.asarray(spMtx_csr.data, dtype=cp.float32), cp.asarray(spMtx_csr.indices), cp.asarray(spMtx_csr.indptr)),
				shape=spMtx_csr.shape
		)

		# Compute quInterest and its norm
		quInterest = query_vec_squeezed * idf_squeezed
		quInterestNorm = cp.linalg.norm(quInterest)

		# Get indices of non-zero elements in quInterest
		idx_nonzeros = cp.nonzero(quInterest)[0]
		quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm

		# Normalize user interests
		usrInterestNorm = spMtx_norm + cp.float32(1e-4)

		# Initialize result array
		cs = cp.zeros(spMtx_gpu.shape[0], dtype=cp.float32)

		# Process in batches to avoid memory overflow
		for i in range(0, spMtx_gpu.shape[0], batch_size):
				# Define batch range
				start_idx = i
				end_idx = min(i + batch_size, spMtx_gpu.shape[0])

				# Extract batch from sparse matrix
				spMtx_batch = spMtx_gpu[start_idx:end_idx, :]

				# Extract only the necessary columns from the batch
				spMtx_nonZeros = spMtx_batch[:, idx_nonzeros]

				# Apply IDF and normalize
				spMtx_nonZeros = spMtx_nonZeros.multiply(idf_squeezed[idx_nonzeros])
				spMtx_nonZeros = spMtx_nonZeros.multiply(1 / usrInterestNorm[start_idx:end_idx, None])

				# Apply exponent if necessary
				if exponent != 1.0:
						spMtx_nonZeros.data **= exponent

				# Compute cosine similarity scores for the batch
				cs_batch = spMtx_nonZeros.dot(quInterest_nonZeros)

				# Store batch results
				cs[start_idx:end_idx] = cs_batch

				# Free memory for the batch
				del spMtx_batch, spMtx_nonZeros, cs_batch
				cp.get_default_memory_pool().free_all_blocks()
				torch.cuda.empty_cache() # Clear CUDA cache
				# torch.cuda.synchronize() # Ensure all CUDA operations are complete
				# Print memory usage after each batch
				# print(f"Batch {i // batch_size + 1}: Free GPU Memory: {device.mem_info[0] / 1024 ** 3:.2f} GB")

		print(f"Elapsed_t: {time.time() - st_t:.2f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, " "))
		return cp.asnumpy(cs)  # Convert result back to NumPy for compatibility

def get_customized_recsys_avg_vec_gpu(spMtx, cosine_sim, idf_vec, spMtx_norm, batch_size:int=2048):
		print(f"[GPU optimized] avgRecSys (1 x nTKs={spMtx.shape[1]})".center(130, "-"))
		st_t = time.time()
		
		# Move data to GPU
		idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
		cosine_sim_gpu = cp.asarray(cosine_sim, dtype=cp.float32)
		spMtx_norm_gpu = cp.asarray(spMtx_norm, dtype=cp.float32)
		
		# Find non-zero cosine similarities
		non_zero_cosines = cp.nonzero(cosine_sim_gpu)[0]
		non_zero_values = cosine_sim_gpu[non_zero_cosines]
		
		print(
				f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
				f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
				f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {len(non_zero_cosines)}\n"
				f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
		)
		
		# Convert sparse matrix to CuPy CSR format
		spMtx_csr = spMtx.tocsr()
		spMtx_gpu = cp.sparse.csr_matrix(
				(cp.asarray(spMtx_csr.data, dtype=cp.float32),
				 cp.asarray(spMtx_csr.indices),
				 cp.asarray(spMtx_csr.indptr)),
				shape=spMtx_csr.shape
		)
		
		# Initialize result array on GPU
		avg_rec = cp.zeros(spMtx.shape[1], dtype=cp.float32)
		
		# Process in batches
		for i in range(0, len(non_zero_cosines), batch_size):
				batch_indices = non_zero_cosines[i:i + batch_size]
				batch_values = non_zero_values[i:i + batch_size]
				
				# Extract batch from sparse matrix
				spMtx_batch = spMtx_gpu[batch_indices]
				
				# Apply IDF
				batch_result = spMtx_batch.multiply(idf_squeezed)
				
				# Normalize by user interest norm
				norm_factors = spMtx_norm_gpu[batch_indices] + cp.float32(1e-18)
				batch_result = batch_result.multiply(1.0 / norm_factors[:, None])
				
				# Multiply by cosine similarities
				batch_result = batch_result.multiply(batch_values[:, None])
				
				# Add to running sum
				avg_rec += batch_result.sum(axis=0).ravel()
				
				# Clean up memory
				del batch_result, spMtx_batch
				cp.get_default_memory_pool().free_all_blocks()
		
		# Normalize the result
		avg_rec /= cp.sum(non_zero_values)
		
		# Convert back to CPU
		result = cp.asnumpy(avg_rec)
		
		print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(result)} {result.dtype} {result.shape}".center(130, " "))
		return result


:root {
	--primary-color: #5b37b1;
	--spacing-unit: 8px;
}

body {
	display: flex;
	flex-direction: column;
	height: 100vh;
	max-width: 1200px; /* Ensure max-width for consistency on all screens */
	margin: 0 auto; /* centered layout */
	padding: 0;
	box-sizing: border-box;
}

@keyframes glow {
	from {
		text-shadow: 
			0 0 10px #5827cab0, 
			0 0 20px #ebff78, 
			0 0 40px #edf5c8, 
			0 0 80px #d4df98;
	}
	
	to {
		text-shadow: 
			0 0 10px rgb(50, 180, 180), 
			0 0 30px #81adf0, 
			0 0 50px #c699f0, 
			0 0 50px #dbbacb,
			0 0 90px #e4d2db;
	}
}

.glow {
	color: #6b47c0b0;
	font-weight: bold;
	font-family: 'Poppins', sans-serif;
	text-align: center;
	font-size: 2rem;
	margin-top: 1rem;
	margin-bottom: 1rem;
	animation: glow 0.85s ease-in-out infinite alternate; /* Slows the animation */
}

.navbar {
	display: flex;
	align-items: center;
	justify-content: flex-start;
	background-color: #333;
	padding: 10px 20px;
	border-radius: 0 0 8px 8px;
}

.navbar a {
	text-decoration: none;
	color: white;
	font-size: 1.15rem;
	padding: 12px 20px;
	margin-right: 15px;
	transition: background-color 0.3s ease; /* Smooth transition */
}

.navbar a.home {
	background-color: #5b37b1;
	font-weight: bold;
	border-radius: 4px;
}

.navbar a:hover {
	background-color: #575757;
	color: white;
}

.container {
	text-align: center;
	transition: transform 0.5s ease, width 0.5s ease;
	width: 100%; /* Ensure full width for proper translation */
}

.container.translated {
	transform: translateX(-30%);
	width: 70%;
}

.imageContainer {
	display: flex;
	justify-content: center;
}

.imageContainer img {
	width: 100%; /* Full width, will scale naturally */
	max-width: 200px; /* Prevents images from growing too large */
	height: auto; /* Maintain aspect ratio */
	margin: 10px;
	filter: grayscale(100%);
}

.search-container {
	width: 100%;
	justify-content: center;
	/* background-color: #ffdac1d8; */
	align-items: center;
}

.search-container h2 {
	font-size: 25px;
	font-weight: bold;
	color: #000;
	margin-bottom: 5px;
}

.search-container h3 {
	color: #000;
	margin-bottom: 20px;
}

.search-form {
	justify-content: center;
	position: relative;
	height: 380px;
}

.search-form::before {
	content: "";
	position: absolute;
	top: 0;
	right: 0;
	bottom: 0;
	left: 0;
	background-image: url("https://aptitude-test.com/wp-content/uploads/2023/05/pic.jpg");
	background-size: 64% 100%;
	background-repeat: no-repeat;
	background-position: center top 0px;
	filter: grayscale(0.8);
	z-index: -1;
}

.search-input {
	position: relative;
}

.search-input-field {
	width: 60%; /* Fixed width that works well on different screen sizes */
	height: 25px;
	font-size: 1.35rem;
	padding: 10px;
	font-weight: bold;
	font-family: Georgia, 'Times New Roman', Times, serif;
	border-radius: 8px;
	border: none;
	background-color: #e1e2e2;
	margin-top: 18px;
	caret-color: rgb(26, 250, 201);
}

.search-input-field:focus {
	background-color: #ffffff;
	color: #303030c5;
	border: 2px solid #080808;
}

.help-container {
	width: 32%;
	height: 75%;
	font-size: 16px;
	font-weight: bold;
	position: absolute;
	top: 0;
	right: 0;
	background: transparent url("https://i.pinimg.com/564x/1c/f7/80/1cf7809521b1fc112c8b116ccb1e2a01.jpg") no-repeat scroll center;
	background-size: 180px 50px;
	display: flex;
	justify-content: center;
	align-items: center;
	display: none; /* Initially hidden */
}

.search-input-field:focus + .help-container {
	display: flex;
	justify-content: center;
	align-items: center;
	text-decoration: none;
	z-index: 1;
}

.fold {
	width: 45%;
	height: 300px;
	border-radius: 15px;
	color: #0c0c0cc4;
	position: absolute;
	left: calc(54% + 0px);
	top: calc(80% + 0px);
	text-align: left;
	padding: 10px;
	background: -webkit-linear-gradient(top, #e6e6e6e7, #d1b5fd93);
	font-size: 15px;
	font-family: Cambria, Cochin, Georgia, Times, 'Times New Roman', serif;
	transition: all 0.7s ease;
}

.unfolder { 
	display: none;
}

.toggle-label {
	display: inline-block;
	cursor: pointer;
}

.unfold-icon, .fold-icon {
	color: #7b47db;
	width: 10px;
	display: inline-block;
}

.unfolder ~ .fold {
	display: none;
}

.unfolder ~ label .fold-icon {
	display: none;
}

.unfolder:checked ~ .fold {
	display: block;
}

.unfolder:checked ~ label .fold-icon {
	display: inline-block;
}

.unfolder:checked ~ label .unfold-icon {
	display: none;
}

.button-container {
	display: flex;
	justify-content: center;
	align-items: center;
	gap: 25px;
	margin: 25px;
}

.button-container.vertical {
	flex-direction: column;
}

.btn-nlf-search:hover {
	background-color: rgba(0, 3, 200, 0.8);
	color: #e2e0e0;
	font-weight: bold;
}

.btn-clear:hover {
	background-color: rgba(200, 1, 0, 0.8);
	color: #e2e0e0;
	font-weight: bold;
}

input[type="submit"]:hover {
	background-color: rgba(5, 116, 8, 0.8);
	color: #e2e0e0;
	font-weight: bold;
}

.btn-nlf-search, .btn-clear, input[type="submit"] {
	width: clamp(100px, 10vw, 150px);
	height: 35px;
	font-size: 18px;
	font-weight: normal;
	border-radius: 15px;
	margin: 2px 0;
	transition: all 0.3s ease;
	background-color: rgb(149, 145, 145);
	color:#000;
	border: none;
	outline: none;
	cursor: pointer;
	font-family: 'Times New Roman', Times, serif;
}

#libraryLinkContainer {
	font-size: 22px;
	font-weight: bold;
	color: rgb(1, 10, 250);
	font-family: 'Times New Roman', Times, serif;
}

.blur-background {
	backdrop-filter: invert(80%);
	display: inline-block;
	padding: 10px;
	background-color: #9b9b9bc7;
}

.slider-container {
	margin: 10px;
	text-align: center;
}

#recSysSlider {
	-webkit-appearance: none;
	appearance: none;
	width: 32%;
	height: 10px;
	background: rgb(180, 180, 180);
	outline: none;
	opacity: 0.4;
	-webkit-transition: .4s;
	transition: opacity .4s;
	border-radius: 9px;
}

#recSysSliderLbl {
	font-size: 16px;
	font-family: 'Times New Roman', Times, serif;
	font-style: oblique;
	background-color: #f0ec00;
}

#recSysSlider:hover {
	opacity: 1;
}

#recSysSlider::-webkit-slider-thumb {
	-webkit-appearance: none;
	appearance: none;
	width: 20px;
	height: 20px;
	background: #07880c;
	cursor: pointer;
	border-radius: 50%;
}

#recSysSlider::-moz-range-thumb {
	width: 20px;
	height: 20px;
	background: #4CAF50;
	cursor: pointer;
	border-radius: 50%;
}

.loading-container {
	display: flex;
	flex-direction: column;
	align-items: center;	
}

.spinner-text {
	color: #ffffffe8;
	font-family: 'Times New Roman', Times, serif;
	font-size: 25px;
	font-weight: bold;
	font-style: oblique;
	backdrop-filter: blur(5px);
	background-color: rgba(179, 179, 179, 0.644);
	-webkit-backdrop-filter: blur(5px);
}

.loading-spinner {
	display: none;
}

.loading-spinner:before {
	content: '';
	box-sizing: border-box;
	position: absolute;
	width: 65px;
	height: 65px;
	margin-left: -70px;
	border-radius: 70%;
	border: 5px solid #e0e0e0;
	border-top: 1px solid transparent;
	animation: spin 0.7s linear infinite;
}

@keyframes spin {
	0% { transform: rotate(0deg); }
	100% { transform: rotate(360deg); }
}

.feedback-option, .feedback-label {
	width: 100px; /* Set a fixed width for both columns */
	text-align: center; /* Center-align the content */
	font-weight: bold; /* Make the text bold */
}

/* Recommendation Table CSS layout */
/* ##################################### */
.recommendationsContainer {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	background-color: #ffffff; /* MUST BE SET TO WHITE*/
}

#recSysIntroContainer {
	color: rgba(1, 20, 14, 0.6);
	font-weight: bold;
	font-size: 22px;
}

#recommendationTable {
	width: 100%; /* Keep the table consistent in width */
	border-collapse: collapse; /* Collapse borders between cells */
	margin: 0 auto;
	font-family: Cambria, Cochin, Georgia, Times, 'Times New Roman', serif;
}

#recommendationTable th {
	background-color: #222222;
	color: white;
	padding: 8px;
	text-align: center;
	position: sticky; /* Make the header sticky */
	top: 0; /* Stick to the top of the container */
	z-index: 2; /* Ensure it stays above other content */
}

#recommendationTable tr {
	font-size: 25px;
	background-color: #bebebebe; /* Light gray background for rows */
}

#recommendationTable tr:nth-child(even) {
	background-color: #747474ab; /* Darker gray for even rows */
}

#recommendationTable td {
	padding: 41px; /* must be padding for adjuctment of text, box and chart*/
	border: 1px solid #dadada; /* Light gray border around cells */
	text-align: left;
}

.rec-link {
	display: inline-block;
	vertical-align: middle;
	text-align: left;
	transition: all 0.3s ease;
}

/* 
#####For any presentations, it should be uncommented!#####
#recommendationResultContainer tr:hover .rec-link {
	font-size: 1.15em;
	line-height: 2.8;
	background-color: rgba(223, 223, 223, 0.93);
	color: #001cb9;
	padding: 1px;
	border-radius: 5px;
	position: relative;
	z-index: 1;
}

#recommendationResultContainer tr:hover .rec-link::before {
	content: "";
	position: absolute;
	top: 0;
	left: 0;
	right: 0;
	bottom: 0;
	background: rgba(180, 180, 180, 0.815);
	filter: blur(8px);
	z-index: -1;
	border-radius: 8px;
}

#recommendationResultContainer tr:hover {
	background-color: inherit;
} 
#####For any presentations, it should be uncommented!#####
*/

tbody tr {
	position: relative; /* table row position is relative */
}

td:first-child {
	position: relative;
	padding-right: 50px;
	text-align: left;
}

.pie-chart-container {
	display: inline-block;
	width: 90px;
	height: 90px;
	position: absolute;
	top: 50%;
	transform: translateY(-50%);
	right: 120px; /* Position it to the left of the circular box */
}

.pieChart {
	width: 100%;
	height: 100%;
}

.pieSegment {
	transition: transform 0.2s;
	transform-origin: center;
	/* stroke: black;
	stroke-width: 1px; */
}

.pieSegment:hover {
	transform: scale(1.02);
	filter: brightness(1.09);
	/* stroke-width: 2px; */
}

#recommendationResultContainer tr:hover .pie-chart-container {
	transform: translateY(-50%) scale(1.6);
	transition: transform 0.3s ease-in-out;
}

.pie-chart-legend-container {
	display: flex;
	align-items: center;
	justify-content: center;
	margin-top: 10px;
	color: #383838;
	/* font-style: oblique; */
}

.legend-container {
	display: flex;
	justify-content: center;
}

.legend-item {
	display: flex;
	align-items: center;
	margin: 0 15px;
}

.legend-color {
	width: 25px;
	height: 25px;
	border-radius: 50%;
	margin-right: 5px;
}

.legend-text {
	color: #e9e9e9;
	font-weight: bold;
	font-size: 17px;
}


/* Responsive scaling based on screen size using media queries */
@media screen and (max-width: 768px) {
	.button-container.vertical {
			right: -100px;
			top: 40%;
	}

	/* .button-container {
		gap: 15px;
	} */

	.btn-nlf-search, .btn-clear, input[type="submit"] {
			width: 100px;
			font-size: 14px;
	}
}

@media screen and (max-width: 480px) {
	.button-container.vertical {
			right: -80px;
			top: 30%;
	}
	
	.btn-nlf-search, .btn-clear, input[type="submit"] {
			width: 80px;
			font-size: 12px;
	}
}




/* Responsive scaling based on screen size using media queries */
@media screen and (max-width: 1200px) {
	.body {
		max-width: 100%;
		padding: 0 15px; /* Add padding for smaller screens */
	}
}

@media screen and (max-width: 768px) {
	.navbar a {
		font-size: 1.6rem;
	}


	/* .btn-nlf-search, .btn-clear, input[type="submit"] {
		width: 120px;
		height: 35px;
		font-size: 1rem;
		transition: all 0.3s ease;
	} */

	.pie-chart-container {
		width: 60px;
		height: 60px;
		right: 50px;
	}
}

@media screen and (max-width: 480px) {
	.search-input-field {
		width: 80%; /* Expand the input for smaller screens */
	}

	.pie-chart-container {
		width: 50px;
		height: 50px;
		right: 30px;
	}
}

.circular-box {
	display: inline-block;
	width: 75px;
	height: 40px;
	line-height: 40px;
	border-radius: 8%;
	background-color: #021064;
	color: #ffffff;
	font-size: 18px;
	font-weight: bold;
	text-align: center;
	position: absolute;
	top: 50%;
	transform: translateY(-50%);
	right: 10px;
}