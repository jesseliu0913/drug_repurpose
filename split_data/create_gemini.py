# import os
# import json
# import torch
# import json
# import time

# import argparse
# import jsonlines

# import google.genai.errors
# import pandas as pd
# from google import genai
# from google.genai import types
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# token = os.getenv("HF_TOKEN")
# MAX_RETRIES = 100

# def call_gemini(input_text):
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = client.models.generate_content(
#                 model="gemini-2.0-flash",
#                 contents=input_text,
#                 config=types.GenerateContentConfig(
#                     temperature=0.2,
#                     max_output_tokens=1000,
#                     top_p=0.8,
#                     top_k=40,
#                     candidate_count=1,
#                     stop_sequences=["END"],
#                     presence_penalty=0.2,
#                     frequency_penalty=0.3,
#                     seed=42,
#                 )
#             )
#             time.sleep(5)
#             return response.text
#         except google.genai.errors.ClientError as e:
#             if 'RESOURCE_EXHAUSTED' in str(e) and attempt < MAX_RETRIES - 1:
#                 print(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt + 1})")
#                 time.sleep(5)
#             else:
#                 raise

# ddphenotype_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/ddphenotype.csv")
    
# client = genai.Client(api_key='AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I')
# file_path = f"./data/ddphenotype.jsonl"

# with jsonlines.open(file_path, "a") as f_write:
#     for index, row in ddphenotype_data.iterrows():
#         drug_name = row.drug_name
#         disease_name = row.disease_name
#         phenotype_name = row.phenotype_name
#         input_text = f"Given that {disease_name} is associated with {phenotype_name}, which is treatable by {drug_name}, it follows that {disease_name} is an indication for {drug_name}."

#         print(input_text)
#         # answer = call_gemini(input_text)
#         line_dict = {}
#         break
#         # f_write.write(line_dict)
    
