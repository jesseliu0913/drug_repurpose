import os
import json
import torch
import json
import time

import argparse
import jsonlines

import google.genai.errors
import pandas as pd
from google import genai
from google.genai import types
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser(description="Baseline Experiments")
parser.add_argument("--output_path", type=str, help="Input output path")
parser.add_argument("--prompt_type", type=str, help="Input the Prompt Type (raw, cot, phenotype, gene...)")
parser.add_argument("--shuffle_num", type=int, help="For one question, shufflue x times")
args = parser.parse_args()

MAX_RETRIES = 100

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
    
client = genai.Client(api_key='AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I')
num_smaples = args.shuffle_num

os.makedirs(args.output_path, exist_ok=True)
prompt_type = args.prompt_type
file_path = f"{args.output_path}/{prompt_type}_{num_smaples}.jsonl"

disease_keys = list(disease_data.keys())[8:400]
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
            if prompt_type == "phenotype":
                prefix = f"{dk} includes the following phenotypes: {phenotype}"
                question = f"{prefix}\nIs {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} directly answer me with YES or NO\nAnswer:"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
            elif prompt_type == "cot":  
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} let's think step by step and then answer me with YES or NO\nAnswer:"
            elif prompt_type == "fcot":  
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"{two_shot}\nQuestion: {question}."
            elif prompt_type == "gene":  
                prefix = f"{dk} includes the following gene: {gene}"
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} directly answer me with YES or NO\nAnswer:"
            else: 
                question = f"Is {dk} an indication for {drug_pair[0]}?"
                input_text = f"Question: {question} directly answer me with YES or NO\nAnswer:"
            
            answer = call_gemini(input_text)
            answer_lst.append(answer)

        line_dict = {"drug_name": drug_lst, "disease_name": dk, "answer": answer_lst}
        f_write.write(line_dict)
    
