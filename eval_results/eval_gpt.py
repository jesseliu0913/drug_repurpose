import os
from openai import AzureOpenAI
import os
import json
import ast
import argparse
import jsonlines
import pandas as pd
import google.generativeai as genai
import time
import random
from google.api_core import exceptions


parser = argparse.ArgumentParser(description="Baseline Experiments")
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

endpoint = "https://azure-api-jesse.openai.azure.com/"
model_name = "gpt-35-turbo"
deployment = "gpt-35-turbo"

subscription_key = "8H18BGFCgtsRenwiUVLicgiRjVS9PmA44cnOpt2OvvxKgRqIbLMtJQQJ99BDACYeBjFXJ3w3AAABACOGLsu4"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def call_gpt(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant in Drug-Disease relationship task.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        model=deployment
    )

    return response.choices[0].message.content


os.makedirs(args.output_path, exist_ok=True)
prompt_type = args.prompt_type
file_path = f"{args.output_path}/{prompt_type}.jsonl"

test_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
node_data = pd.read_csv("/playpen/jesse/drug_repurpose/PrimeKG/nodes.csv")
existing_pairs = set()
if os.path.exists(file_path):
    with jsonlines.open(file_path, "r") as f_read:
        for line in f_read:
            existing_pairs.add((line["drug_name"], line["disease_name"]))
print("~" * 20)
print(f"Total existing pairs: {len(existing_pairs)}")
with jsonlines.open(file_path, "a") as f_write:
    for index, row in test_data.iterrows():
        drug_name = row.drug_name
        disease_name = row.disease_name
        disease_index = row.disease_index
        relation = row.relation

        if (drug_name, disease_name) in existing_pairs:
            print(f"Skipping {drug_name} - {disease_name}, already processed.")
            continue

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
        
        phenotype = phenotype[:10]
        gene = gene[:10]

        if prompt_type == "phenotype":
            prefix = f"{disease_name} have several phenotypes like: {phenotype}"
            question = f"{prefix}\nIs {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"
        elif prompt_type == "cot":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} let's think step by step and then answer me with $YES$ or $NO$\nREASONING:\nANSWER:"
        elif prompt_type == "fcot":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"{two_shot}\nQuestion: {question}."
        elif prompt_type == "gene":  
            prefix = f"{disease_name} associate with several genes like: {gene}"
            question = f"{prefix}\nIs {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"
        elif prompt_type == "fraw":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"{raw_shot}\nQuestion: {question} directly answer me with $YES$ or $NO$\nANSWER:"
        else: 
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"


        if args.shuffle_num == 1:
            answer = call_gpt(prompt=input_text)
            line_dict = {"drug_name": drug_name, "disease_name": disease_name, "answer": answer, "prompt": input_text, "label": row.relation}
            f_write.write(line_dict)
        else:
            answer_lst = []
            for _ in range(args.shuffle_num):
                answer = call_gpt(prompt=input_text)
                answer_lst.append(answer)
                
            line_dict = {"drug_name": drug_name, "disease_name": disease_name, "answer": answer_lst, "prompt": input_text, "label": row.relation}
            f_write.write(line_dict)
