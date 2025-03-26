import os
import json
import torch
import json
import argparse
import jsonlines

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser(description="Baseline Experiments")
parser.add_argument("--model_name", type=str, help="Model Name")
parser.add_argument("--output_path", type=str, help="Input output path")
parser.add_argument("--prompt_type", type=str, help="Input the Prompt Type (raw, cot, phenotype, gene...)")
parser.add_argument("--shuffle_num", type=int, help="For one question, shufflue x times")
args = parser.parse_args()

df = pd.read_excel('../PharmacotherapyDB/catalog.xlsx')  
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", use_auth_token=token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
"""
DM -- disease modifying
SYM -- symptomatic
NOT -- non-indication
"""
cate_dict = {"DM": "disease modifying", "SYM": "symptomatic", "NOT": "non-indication"}
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
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
        
      inputs = tokenizer(input_text, return_tensors="pt").to(device)
      output = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.2)
      answer = tokenizer.decode(output[0], skip_special_tokens=True)
      answer = answer.replace(input_text, "").strip()

      line_dict = {"drug_name": drug, "disease_name": disease, "answer": answer}
      f_write.write(line_dict)
    
