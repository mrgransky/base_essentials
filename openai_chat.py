from openai import OpenAI
import pandas as pd
from tqdm import tqdm
# deepseek-r1, deepseek-v3, gpt-3.5-turbo, gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-4.1-nano, gpt-4.1, gpt-5-mini, gpt-5-nano, gpt-5
model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# model = "google/gemma-3-27b-it"
# model = "mistralai/Mistral-Nemo-Instruct-2407"
base_url = 'https://api.chatanywhere.tech/v1'
# client = OpenAI(
# 	api_key='sk-7XLhcZIoNzEBNM7muHZ5bSU8CgK0ViTKWad6kfWCC8FjF9yP',  # Get this from the project's site
# 	base_url=base_url,
# )
client = OpenAI(
		api_key="8z11Uq1Fm1OMGhV4HpNkAKDkKWHdOZqJ",
		base_url="https://api.deepinfra.com/v1/openai",
)

LLM_INSTRUCTION_TEMPLATE = """<s>[INST]
Act as a meticulous historical archivist specializing in 20th century documentation.
Given the description below, extract up to {k} most prominent, factual and distinct **KEYWORDS** that appear in the text.

{description}

**CRITICAL RULES**:
- Extract **ONLY keywords that actually appear** in the description above.
- Return **AT MOST {k} keywords** - fewer is acceptable if the description is short or lacks distinct concepts.
- Return **ONLY** a clean, valid and parsable **Python LIST** with a maximum of {k} keywords.
- **ABSOLUTELY NO** additional explanatory text, code blocks, terms containing numbers, comments, tags, thoughts, questions, or explanations before or after the **Python LIST**.
- **STRICTLY EXCLUDE ALL TEMPORAL EXPRESSIONS**: No dates, times, time periods, seasons, months, days, years, decades, centuries, or any time-related phrases (e.g., "early evening", "morning", "20th century", "1950s", "weekend", "May 25th", "July 10").
- **STRICTLY EXCLUDE** vague, generic, or historical keywords.
- **STRICTLY EXCLUDE** special characters, stopwords, meaningless, repeating or synonym-duplicate keywords.
- The parsable **Python LIST** must be the **VERY LAST THING** in your response.
[/INST]
"""
df = pd.read_csv(filepath_or_buffer="/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv", on_bad_lines='skip', low_memory=False)

description = "F 4 aircraft are shipped to Hammarnäset J 29. July 10, 1969."
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
			'content': LLM_INSTRUCTION_TEMPLATE.format(k=5, description=description)
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
