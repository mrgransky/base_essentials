from openai import OpenAI
import pandas as pd
from tqdm import tqdm

# chatanywhere:
# deepseek-r1, deepseek-v3, gpt-3.5-turbo, gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-4.1-nano, gpt-4.1, gpt-5-mini, gpt-5-nano, gpt-5
# chatanywhere_token = os.getenv("CHATANYWHERE_API_KEY")
# client = OpenAI(
# 	api_key=chatanywhere_token,  # Get this from the project's site
# 	base_url="https://api.chatanywhere.tech/v1",
# )

# deepinfra:
model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# model = "google/gemma-3-27b-it"
# model = "mistralai/Mistral-Nemo-Instruct-2407"
deepinfra_token = os.getenv("CHATANYWHERE_API_KEY")
client = OpenAI(
	api_key=deepinfra_token,
	base_url="https://api.deepinfra.com/v1/openai",
)

LLM_INSTRUCTION_TEMPLATE = """<s>[INST]
You function as a historical archivist whose expertise lies in the 20th century.
Given the image caption below, extract no more than {k} highly prominent, factual and distinct **KEYWORDS** that convey the primary actions, objects, or occurrences.

{caption}

STRICT RULES — follow exactly:
- Return **ONLY** a clean, valid, and parsable **Python LIST** with **AT MOST {k} KEYWORDS** - fewer is expected if the text is either short or lacks distinct concepts.
- Extract **ONLY** self-contained and grammatically complete phrases that actually appear in the text.
- AVOID incomplete fragments that start or end with prepositions or conjunctions.
- **PRIORITIZE MEANINGFUL PHRASES**: Opt for multi-word n-grams such as NOUN PHRASES and NAMED ENTITIES over single terms only if they convey a more distinct meaning.
- **STRICTLY EXCLUDE NUMERICAL CONTENT** such as numerical values, measurements, units, or quantitative terms.
- **STRICTLY EXCLUDE MEDIA DESCRIPTORS** such as generic photography, image, picture, or media terms.
- **STRICTLY EXCLUDE TEMPORAL EXPRESSIONS** such as specific times, calendar dates, seasonal periods, or extended historical eras.
- **ABSOLUTELY NO** synonymous, duplicate, identical or misspelled keywords.
- **ABSOLUTELY NO** additional explanatory text, code blocks, comments, tags, thoughts, questions, or explanations before or after the **Python LIST**.
- The parsable **Python LIST** must be the **VERY LAST THING** in your response.
[/INST]
"""

df = pd.read_csv(filepath_or_buffer="/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv", on_bad_lines='skip', low_memory=False)

description = """
Ju 88A-17. Ju 88A-17 MTO 1943/44. Junkers Ju 88.
"""
# descriptions = df['enriched_document_description'].tolist()

if isinstance(description, str):
	descriptions = [description]

print(f"Loaded {len(descriptions)} {type(descriptions)} descriptions")
for i, description in tqdm(enumerate(descriptions), total=len(descriptions), desc="Processing descriptions", ncols=100):
	completion = client.chat.completions.create(
		model=model,
		messages=[
		{
			"role": "system", 
			"content": "You are a helpful assistant."
		},
		{
			'role': 'user',
			'content': LLM_INSTRUCTION_TEMPLATE.format(k=5, caption=description)
		}
	],
	temperature=0.0,
	stream=False, # whether to stream the results or not
	)
	# print("-"*100)
	# print(completion)
	# print("-"*100)
	print(f"\nDescription {i}: {description}")
	print(completion.choices[0].message.content)
	print(completion.usage.prompt_tokens, completion.usage.completion_tokens)
	print("-"*100)

# # Assume openai>=1.0.0
# from openai import OpenAI

# # Create an OpenAI client with your deepinfra token and endpoint
# openai = OpenAI(
#     api_key="8z11Uq1Fm1OMGhV4HpNkAKDkKWHdOZqJ",
#     base_url="https://api.deepinfra.com/v1/openai",
# )

# chat_completion = openai.chat.completions.create(
#     model="Qwen/Qwen3-235B-A22B-Instruct-2507",
#     messages=[{"role": "user", "content": "Hello"}],
# )

# print(chat_completion.choices[0].message.content)
# print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

# # Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?
# # 11 25
