import os
import json
import torch
import argparse
import jsonlines

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


parser = argparse.ArgumentParser(description="Baseline Experiments")
parser.add_argument("--model_name", type=str, help="Model Name")
parser.add_argument("--output_path", type=str, help="Input output path")
parser.add_argument("--shuffle_num", type=int, help="For one question, shufflue x times")
args = parser.parse_args()

df = pd.read_csv('../drug_data/unique_drug_disease.csv')
df_drug = df[df["target_type"] == "disease"]

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
num_smaples = args.shuffle_num

os.makedirs(args.output_path, exist_ok=True)
file_path = f"{args.output_path}/output_DD1k_{num_smaples}.jsonl"

with jsonlines.open(file_path, "a") as f_write:
  for dd_pair in df_drug.itertuples(index=False):
    answer_lst = []
    line_dict = {}

    drug_name = dd_pair.drug_name
    target_name = dd_pair.target_name

    question = f"Whether {drug_name} can treat {target_name}?"
    input_text = f"Question: {question} Directly answer me with yes or no.\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    for _ in range(num_smaples):
      output = model.generate(**inputs, max_new_tokens=10, do_sample=True, top_k=50, temperature=0.8)
      answer = tokenizer.decode(output[0], skip_special_tokens=True)
      answer = answer.replace(input_text, "").strip()
      answer_lst.append(answer)

    line_dict = {"drug_name": drug_name, "target_name": target_name, "effect_type": dd_pair.effect_type, "answer": answer_lst}
    f_write.write(line_dict)
  
    