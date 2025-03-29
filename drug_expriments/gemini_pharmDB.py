import os
import json
import torch
import json
import time
import argparse
import jsonlines

import google.genai.errors
from google import genai
from google.genai import types
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser(description="Baseline Experiments")
parser.add_argument("--output_path", type=str, help="Input output path")
parser.add_argument("--prompt_type", type=str, help="Input the Prompt Type (raw, cot, phenotype, gene...)")
parser.add_argument("--shuffle_num", type=int, help="For one question, shufflue x times")
args = parser.parse_args()


MAX_RETRIES = 100
client = genai.Client(api_key='AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I')

def call_gemini(input_text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=input_text,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                    top_p=0.8,
                    top_k=40,
                    candidate_count=1,
                    stop_sequences=["END"],
                    presence_penalty=0.2,
                    frequency_penalty=0.3,
                    seed=42,
                )
            )
            time.sleep(5)
            return response.text
        except google.genai.errors.ClientError as e:
            if 'RESOURCE_EXHAUSTED' in str(e) and attempt < MAX_RETRIES - 1:
                print(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt + 1})")
                time.sleep(5)
            else:
                raise

df = pd.read_excel('../PharmacotherapyDB/catalog.xlsx')  
"""
DM -- disease modifying
SYM -- symptomatic
NOT -- non-indication
"""
cate_dict = {"DM": "disease modifying", "SYM": "symptomatic", "NOT": "non-indication"}
device = "cuda" if torch.cuda.is_available() else "cpu"

num_smaples = args.shuffle_num

os.makedirs(args.output_path, exist_ok=True)
prompt_type = args.prompt_type
file_path = f"{args.output_path}/{prompt_type}_{num_smaples}.jsonl"

with jsonlines.open(file_path, "a") as f_write:
  for index, row in df.iterrows():
      line_dict = {}
      disease = row['disease']
      drug = row['drug']
      category = cate_dict.get(row['category'])

      question = f"What is the relationship between {disease} and {drug}?\nA.disease modifying \nB.symptomatic\nC.non-indication "
      if prompt_type == "cot":
        input_text = f"Question: {question} \nLet's think step by step and then answer me\nAnswer:"
      else:
        input_text = f"Question: {question} \nDirectly answer me\nAnswer:"
        
      answer = call_gemini(input_text)
      answer = answer.strip()

      line_dict = {"drug_name": drug, "disease_name": disease, "answer": answer}
      f_write.write(line_dict)
    
