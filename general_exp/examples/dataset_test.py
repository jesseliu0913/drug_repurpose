import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# dict_keys(['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer'])
ds = load_dataset("mandarjoshi/trivia_qa", "rc")
sub_test = ds['validation'].select(range(500))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for qa_pair in sub_test:
  groundtruth = qa_pair['answer']['value']
  question = qa_pair['question']
  input_text = f"Question: {question} Directly answer me without any other words.\nAnswer:"
  inputs = tokenizer(input_text, return_tensors="pt").to(device)
  output = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.1, temperature=0.2)
  answer = tokenizer.decode(output[0], skip_special_tokens=True)
  answer = answer.replace(input_text, "").strip()
  print("Model's Answer:", answer)
  print("groundtruth", groundtruth)
  break
