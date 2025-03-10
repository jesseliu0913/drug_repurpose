import os
import json
import torch
import jsonlines
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


parser = argparse.ArgumentParser(description="Baseline Experiments")
parser.add_argument("--model_name", type=str, help="Model Name")
parser.add_argument("--dataset", type=str, help="Dataset Name")
parser.add_argument("--subset", type=str, help="Subset Name")
parser.add_argument("--output_path", type=str, help="Input output path")
parser.add_argument("--shuffle_num", type=int, help="For one question, shufflue x times")
args = parser.parse_args()

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

os.makedirs(args.output_path, exist_ok=True)
file_path = f"{args.output_path}/output_{args.shuffle_num}.jsonl"
if args.subset != "None":
  ds = load_dataset(args.dataset, args.subset)  
else:
  ds = load_dataset(args.dataset)
  
sub_test = ds['validation']
num_smaples = args.shuffle_num

with jsonlines.open(file_path, "a") as f_write:
  for qa_pair in sub_test:
    answer_lst = []
    line_dict = {}

    # groundtruth = qa_pair['answer']['value']
    groundtruth = qa_pair['answers']
    question = qa_pair['question']
    input_text = f"Question: {question} Directly answer me without any other words.\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    for _ in range(num_smaples):
      output = model.generate(**inputs, max_new_tokens=10, do_sample=True, top_k=50, temperature=0.8)
      answer = tokenizer.decode(output[0], skip_special_tokens=True)
      answer = answer.replace(input_text, "").strip()
      answer_lst.append(answer)

    line_dict = {"question": question, "groundtruth": groundtruth, "answers": answer_lst}
    f_write.write(line_dict)
  
    