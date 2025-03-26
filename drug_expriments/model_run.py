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

two_shot = """
Question: Is Pentamidine an indication for hypertensive disorder?
Reasoning: Pentamidine is contraindicated in patients with hypertensive disorders because it can cause significant cardiovascular side effects, including sudden hypotension, cardiac arrhythmias (such as QT prolongation), and electrolyte disturbances like hypoglycemia and hyperkalemia. These effects can exacerbate underlying cardiovascular instability in hypertensive individuals, increasing the risk of serious complications. Additionally, many patients with hypertension may be on medications that interact adversely with Pentamidine, further compounding the risks.
Answer:$NO$
Question: Is Moxonidine an indication for hypertensive disorder?
Reasoning: Hypertensive disorders are an indication for Moxonidine because it effectively lowers blood pressure by selectively stimulating imidazoline receptors in the brainstem, which reduces sympathetic nervous system activity—a key contributor to high blood pressure. Unlike traditional antihypertensives that may affect heart rate or electrolyte balance, Moxonidine offers targeted central action with fewer metabolic side effects, making it particularly suitable for patients with essential hypertension, especially those with metabolic syndrome or diabetes.
Answer:$YES$
"""

with open("../drug_data/data/disease_feature.json", "r", encoding="utf-8") as file:
    disease_data = json.load(file)
    
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", use_auth_token=token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
num_smaples = args.shuffle_num

os.makedirs(args.output_path, exist_ok=True)
prompt_type = args.prompt_type
file_path = f"{args.output_path}/{prompt_type}_{num_smaples}.jsonl"

disease_keys = list(disease_data.keys())[:400]
with jsonlines.open(file_path, "a") as f_write:
    for dk in disease_keys:
        drug_lst = []
        answer_lst = []
        line_dict = {}
        if "contraindication" in disease_data[dk].keys():
            drug_lst.append([disease_data[dk]['contraindication'][0], "contraindication"])
        if "indication" in disease_data[dk].keys():
            drug_lst.append([disease_data[dk]['indication'][0], "indication"])
        
        if "disease_phenotype_positive" in disease_data[dk].keys():
            phenotype = disease_data[dk]['disease_phenotype_positive']
            phenotype = ",".join(map(str, phenotype))
        
        if "disease_protein" in disease_data[dk].keys():
            gene = disease_data[dk]['disease_protein']
            gene = ",".join(map(str, gene))

        for drug_pair in drug_lst:
            print("prompt_type", prompt_type)
            if prompt_type == "phenotype":
                prefix = f"{dk} includes the following phenotypes: {phenotype}"
                question = f"{prefix}\nIs {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} directly answer me with YES or NO\nAnswer:"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
            elif prompt_type == "cot":  
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} let's think step by step and then answer me with YES or NO\nAnswer:"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
            elif prompt_type == "fcot":  
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"{two_shot}\nQuestion: {question}."
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
            elif prompt_type == "gene":  
                prefix = f"{dk} includes the following gene: {gene}"
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} directly answer me with YES or NO\nAnswer:"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
            else: 
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} directly answer me with YES or NO\nAnswer:"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            output = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.2)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            answer = answer.replace(input_text, "").strip()
            answer_lst.append(answer)

        line_dict = {"drug_name": drug_lst, "disease_name": dk, "answer": answer_lst}
        f_write.write(line_dict)
    
