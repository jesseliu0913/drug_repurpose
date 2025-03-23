import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Based on the information "A revolving door is convenient for two direction travel, but it also serves as a security measure at a secured building or facility."
# Answer the quetsion:

# dict_keys(['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer'])
question = """
Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a secured building or facility.
Choice: 
"bank", "hear sounds", "singing", "liab", "new york"
Answer: """
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_text = f"Question: {question} \nAnswer:"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=1000, do_sample=True, top_p=0.1, temperature=0.8)
