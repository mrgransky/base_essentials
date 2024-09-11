import sys
import os
import time
from textwrap import wrap
import warnings
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import Input
from tensorflow.keras.layers import (
	GRU,
	Add,
	AdditiveAttention,
	Attention,
	Concatenate,
	Dense,
	Embedding,
	LayerNormalization,
	Reshape,
	StringLookup,
	TextVectorization,
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings('ignore') # warnings.simplefilter("ignore")
print(tf.version.VERSION)
tf.config.run_functions_eagerly(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
HOME: str = os.getenv('HOME')
USER: str = os.getenv('USER') # echo $USER
VOCAB_SIZE = 20000  # use fewer words to speed up convergence
ATTENTION_DIM = 512  # size of dense layer in Attention
WORD_EMBEDDING_DIM = 128
print(f">> InceptionResNetV2 model using imageNet weights...")
FEATURE_EXTRACTOR = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
	include_top=False,
	weights="imagenet",
)
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNELS = 3
FEATURES_SHAPE = (8, 8, 1536)

# # Load the coco_captions dataset
# ds, info = tfds.load('coco_captions', split='train', with_info=True)
# print(type(ds),)
# print(type(info))
# print(info)
# print("#"*100)
# # Iterate over the dataset
# for example in ds.take(5):
# 	image = example['image']
# 	captions = example['captions']
# 	print(f"Image ID: {example['image/id']}")
# 	print(f"Captions:")
# 	for caption in captions:
# 		print(f"- {caption['text']}")
# 	# Visualize the image (requires matplotlib)
# 	import matplotlib.pyplot as plt
# 	plt.imshow(image)
# 	plt.show()
# sys.exit()

GCS_DIR = "gs://asl-public/data/tensorflow_datasets/"
BUFFER_SIZE = 1000

if USER == "alijanif":
	customized_dataset_dir = "/scratch/project_2004072/IMG_Captioning"
elif USER == "ubuntu":
	customized_dataset_dir = "/media/volume/IMG_Captioning"
else:
	customized_dataset_dir = f"{os.path.join(HOME, 'tensorflow_datasets')}"

print(f">> Loading CoCo Caption Dataset from/in {customized_dataset_dir}")
t0 = time.time()
trainds = tfds.load(
	name="coco_captions",
	split="train",
	# data_dir=GCS_DIR,
	data_dir=customized_dataset_dir,
	shuffle_files=True,
)
print(f"TRAIN DS: {type(trainds)} Elapsed_t: {time.time()-t0:.2f} sec")
print("#"*100)

def get_image_label(example):
	caption = example["captions"]["text"][0]  # only the first caption per image
	img = example["image"]
	img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
	img = img / 255
	return {"image_tensor": img, "caption": caption}

trainds = trainds.map(
	get_image_label, 
	num_parallel_calls=tf.data.AUTOTUNE,
).shuffle(BUFFER_SIZE)

trainds = trainds.prefetch(buffer_size=tf.data.AUTOTUNE)

# f, ax = plt.subplots(1, 4, figsize=(20, 5))
# for idx, data in enumerate(trainds.take(4)):
# 		ax[idx].imshow(data["image_tensor"].numpy())
# 		caption = "\n".join(wrap(data["caption"].numpy().decode("utf-8"), 30))
# 		ax[idx].set_title(caption)
# 		ax[idx].axis("off")

## Text Preprocessing:
def add_start_end_token(data):
	start = tf.convert_to_tensor("<start>")
	end = tf.convert_to_tensor("<end>")
	data["caption"] = tf.strings.join([start, data["caption"], end], separator=" ",)
	return data

trainds = trainds.map(add_start_end_token)

MAX_CAPTION_LEN = 64

@tf.autograph.experimental.do_not_convert
def standardize(inputs):
	inputs = tf.strings.lower(inputs)
	return tf.strings.regex_replace(
		inputs,
		r"[!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~]?",
		"",
	)

# Choose the most frequent words from the vocabulary & remove stopwords.
tokenizer = TextVectorization(
	max_tokens=VOCAB_SIZE,
	standardize=standardize,
	output_sequence_length=MAX_CAPTION_LEN,
)
print(f">> Adapting tokenizer to the caption [time consuming]...")
t0 = time.time()
tokenizer.adapt(trainds.map(lambda x: x["caption"]))
tokenizer(["<start> This is a sentence <end>"])
print(f"Elapsed_t: {time.time()-t0:.1f} sec")

sample_captions = []
for d in trainds.take(5):
	sample_captions.append(d["caption"].numpy())

print(sample_captions)
print(tokenizer(sample_captions))

for wordid in tokenizer([sample_captions[0]])[0]:
	print(tokenizer.get_vocabulary()[wordid], end=" ")

# Lookup table: Word -> Index
word_to_index = StringLookup(
	mask_token="", 
	vocabulary=tokenizer.get_vocabulary(),
)

# Lookup table: Index -> Word
index_to_word = StringLookup(
	mask_token="", 
	vocabulary=tokenizer.get_vocabulary(), 
	invert=True,
)

BATCH_SIZE = 8
def create_ds_fn(data):
	img_tensor = data["image_tensor"]
	caption = tokenizer(data["caption"])
	target = tf.roll(caption, -1, 0)
	zeros = tf.zeros([1], dtype=tf.int64)
	target = tf.concat((target[:-1], zeros), axis=-1)
	return (img_tensor, caption), target

print(f"Batched Dataset...")
btch_st = time.time()
# batched_ds = trainds.map(create_ds_fn).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
batched_ds = trainds.map(create_ds_fn).batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=tf.data.AUTOTUNE)
print(f"Elapsed_t: {time.time()-btch_st:.1f} sec")

for (img, caption), label in batched_ds.take(2):
	print(f"Image shape: {img.shape}")
	print(f"Caption shape: {caption.shape}")
	print(f"Label shape: {label.shape}")
	print(caption[0])
	print(label[0])

FEATURE_EXTRACTOR.trainable = False

image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
image_features = FEATURE_EXTRACTOR(image_input)
x = Reshape((FEATURES_SHAPE[0] * FEATURES_SHAPE[1], FEATURES_SHAPE[2]))(image_features)
encoder_output = Dense(ATTENTION_DIM, activation="relu")(x)
encoder = tf.keras.Model(inputs=image_input, outputs=encoder_output)
print(encoder.summary())

word_input = Input(
	shape=(MAX_CAPTION_LEN,), 
	name="words",
)
embed_x = Embedding(VOCAB_SIZE, ATTENTION_DIM)(word_input)
decoder_gru = GRU(
	ATTENTION_DIM,
	return_sequences=True,
	return_state=True,
)
gru_output, gru_state = decoder_gru(embed_x)
decoder_attention = Attention()
context_vector = decoder_attention([gru_output, encoder_output])
addition = Add()([gru_output, context_vector])
layer_norm = LayerNormalization(axis=-1)
layer_norm_out = layer_norm(addition)
decoder_output_dense = Dense(VOCAB_SIZE)
decoder_output = decoder_output_dense(layer_norm_out)
decoder = tf.keras.Model(
	inputs=[word_input, encoder_output], 
	outputs=decoder_output,
)
print(decoder.summary())

"""
### Training Model
two inputs (`image_input` and `word_input`, 
output (`decoder_output`).
"""
image_caption_train_model = tf.keras.Model(
	inputs=[image_input, word_input],
	outputs=decoder_output
)
print(image_caption_train_model.summary())

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=True,
	reduction="none",
)
optm = tf.keras.optimizers.Adam()

@tf.function
def loss_function(real, pred):
	loss_ = loss_object(real, pred)
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	mask = tf.cast(mask, dtype=tf.int32)
	sentence_len = tf.reduce_sum(mask)
	loss_ = loss_[:sentence_len]
	return tf.reduce_mean(loss_)

# def mask_and_calculate_length(real):
# 	mask = tf.math.logical_not(tf.math.equal(real, 0))
# 	mask = tf.cast(mask, dtype=tf.int32)
# 	sentence_len = tf.reduce_sum(mask)
# 	return mask, sentence_len

# @tf.function
# def loss_function(real, pred):
# 	loss_ = loss_object(real, pred)
# 	return loss_

# # Calculate mask and sentence length before feeding to loss function
# mask, sentence_len = mask_and_calculate_length(real)

image_caption_train_model.compile(
	optimizer=optm,
	loss=loss_function,
)

print(f"Training...")
train_st_time = time.time()
history = image_caption_train_model.fit(batched_ds, epochs=1)
print(f"Elapsed_t: {time.time()-train_st_time:.2f} sec")

gru_state_input = Input(
	shape=(ATTENTION_DIM,),
	name="gru_state_input",
)
gru_output, gru_state = decoder_gru(embed_x, initial_state=gru_state_input)
context_vector = decoder_attention([gru_output, encoder_output])
addition_output = Add()([gru_output, context_vector])
layer_norm_output = layer_norm(addition_output)
decoder_output = decoder_output_dense(layer_norm_output)

decoder_pred_model = tf.keras.Model(
	inputs=[word_input, gru_state_input, encoder_output],
	outputs=[decoder_output, gru_state],
)

MINIMUM_SENTENCE_LENGTH = 5
def predict_caption(filename):
	gru_state = tf.zeros((1, ATTENTION_DIM))
	img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=IMG_CHANNELS)
	img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
	img = img / 255
	features = encoder(tf.expand_dims(img, axis=0))
	dec_input = tf.expand_dims([word_to_index("<start>")], 1)
	result = []
	for i in range(MAX_CAPTION_LEN):
		try:
			print(f"dec_input shape: {dec_input.shape}")
			print(f"gru_state shape: {gru_state.shape}")
			print(f"features shape: {features.shape}")
			print(f"#"*60)
			predictions, gru_state = decoder_pred_model([dec_input, gru_state, features])
		except Exception as e:
			print(f"Error in iteration {i}: {e}")
			break
		top_probs, top_idxs = tf.math.top_k(input=predictions[0][0], k=10, sorted=False,)
		chosen_id = tf.random.categorical([top_probs], 1)[0].numpy()
		predicted_id = top_idxs.numpy()[chosen_id][0]
		result.append(tokenizer.get_vocabulary()[predicted_id])
		if predicted_id == word_to_index("<end>"):
			return img, result
		dec_input = tf.expand_dims([predicted_id], 1)
	return img, result

# filename = "surf.jpeg"

# if USER == "ubuntu":
# 	# load
# 	filename = "baseball.jpeg"
# 	# filename = "surf.jpeg"
# elif USER == "farid":
# 	filename = os.path.join(HOME, "Bilder", "baseball.jpeg")
# 	# filename = os.path.join(HOME, "Bilder", "surf.jpeg")
# else:
# 	print(f"{USER} not identified!, Exiting...")
# 	sys.exit(1)

filename = "baseball.jpeg"
for i in range(5):
	image, caption = predict_caption(filename)
	print(" ".join(caption[:-1]) + ".")

img = tf.image.decode_jpeg(
	tf.io.read_file(filename), 
	channels=IMG_CHANNELS,
)
# plt.imshow(img)
plt.savefig(f"captioned_{filename}.png")
plt.axis("off")
# plt.clf()