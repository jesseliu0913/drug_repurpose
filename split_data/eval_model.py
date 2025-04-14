import os
import json
import torch
import json
import ast
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
Question: Is Fosinopril an indication for hypertensive disorder?
REASONING: Fosinopril is indicated for hypertensive disorders because it functions as an angiotensin-converting enzyme (ACE) inhibitor, which blocks the conversion of angiotensin I to angiotensin II—a potent vasoconstrictor. By reducing angiotensin II levels, Fosinopril promotes vasodilation, decreases peripheral vascular resistance, and ultimately lowers blood pressure. This mechanism directly addresses the pathophysiology of hypertension, making Fosinopril an effective and commonly prescribed medication for managing high blood pressure and reducing the risk of associated cardiovascular complications.
ANSWER:$YES$
Question: Is Rotigotine an indication for hypertensive disorder?
REASONING: Rotigotine is a dopamine agonist primarily used to treat Parkinson’s disease and restless legs syndrome (RLS). It works by stimulating dopamine receptors in the brain to help manage motor symptoms. While it may have some effects on blood pressure as a side effect (e.g., causing orthostatic hypotension), it is not approved or used as a treatment for hypertension or other hypertensive disorders.
ANSWER:$NO$
"""

raw_shot = """
Question: Is Fosinopril an indication for hypertensive disorder?
ANSWER:$YES$
Question: Is Rotigotine an indication for hypertensive disorder?
ANSWER:$NO$
"""

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", use_auth_token=token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
os.makedirs(args.output_path, exist_ok=True)
prompt_type = args.prompt_type
file_path = f"{args.output_path}/{prompt_type}.jsonl"

test_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/test_data.csv")
node_data = pd.read_csv("/playpen/jesse/drug_repurpose/PrimeKG/nodes.csv")

with jsonlines.open(file_path, "a") as f_write:
    for index, row in test_data.iterrows():
        drug_name = row.drug_name
        disease_name = row.disease_name
        disease_index = row.disease_index
        relation = row.relation
        related_phenotypes = ast.literal_eval(row.related_phenotypes)
        related_proteins = ast.literal_eval(row.related_proteins)
        phenotype = []
        for pheno_index in related_phenotypes:
            node_name = node_data.loc[pheno_index, 'node_name']
            phenotype.append(node_name)

        gene = []
        for gene_index in related_proteins:
            node_name = node_data.loc[gene_index, 'node_name']
            gene.append(node_name)
        
        phenotype = phenotype[:] if len(phenotype) < 10 else phenotype[:10]
        gene = gene[:] if len(gene) < 10 else gene[:10]

        if prompt_type == "phenotype":
            prefix = f"{disease_name} includes the following phenotypes: {phenotype}"
            question = f"{prefix}\nIs {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
        elif prompt_type == "cot":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} let's think step by step and then answer me with $YES$ or $NO$\nREASONING:\nANSWER:"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
        elif prompt_type == "fcot":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"{two_shot}\nQuestion: {question}."
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
        elif prompt_type == "gene":  
            prefix = f"{disease_name} includes the following genes: {gene}"
            question = f"{prefix}\nIs {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
        elif prompt_type == "fraw":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"{raw_shot}\nQuestion: {question} directly answer me with $YES$ or $NO$\nANSWER:"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
        else: 
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

        output = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.2)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = answer.replace(input_text, "").strip()

        line_dict = {"drug_name": drug_name, "disease_name": disease_name, "answer": answer, "prompt": input_text}
        f_write.write(line_dict)
