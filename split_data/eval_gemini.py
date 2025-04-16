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

# genai.configure(api_key="AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI")
# AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI
# AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc
# AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I
# AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw

api_keys = [
    "AIzaSyAsmMUeXmkOKwjmx__-rhZhyCevd5gllFc", 
    "AIzaSyDK2hSlAMZsrqFBSMR8C2cKW6-u9xCXato",
    "AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI",
    "AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc",
    "AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I",
    "AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw",
]

def call_gemini(message, api_keys, temperature=0.7, max_output_tokens=1000, top_p=0.9, max_retries=5, initial_delay=2):
    if not api_keys:
        raise ValueError("No API keys provided")
    
    rate_limited_keys = set()
    
    while len(rate_limited_keys) < len(api_keys):
        available_keys = [key for key in api_keys if key not in rate_limited_keys]
        if not available_keys:
            print(f"All keys are rate limited. Waiting for 30 seconds before retrying...")
            time.sleep(30)
            rate_limited_keys.clear()
            continue
        
        current_key = available_keys[0]
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        for attempt in range(max_retries):
            try:
                print(f"Using key {api_keys.index(current_key) + 1}/{len(api_keys)} (Attempt {attempt + 1}/{max_retries})")
                
                response = model.generate_content(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": message}]
                        }
                    ],
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "top_p": top_p
                    }
                )
                
                return response.text
                
            except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, 
                    exceptions.TooManyRequests, exceptions.DeadlineExceeded) as e:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit for key {api_keys.index(current_key) + 1}. "
                      f"Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                
                if attempt == max_retries - 1:
                    print(f"Key {api_keys.index(current_key) + 1} is rate limited. Switching to next key.")
                    rate_limited_keys.add(current_key)
                    break
            
            except Exception as e:
                print(f"Unexpected error with key {api_keys.index(current_key) + 1}: {e}")
                rate_limited_keys.add(current_key)
                break
    
    raise Exception("All API keys have been rate limited or encountered errors. Please try again later.")

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

        answer = call_gemini(
            message=input_text,
            api_keys=api_keys,
            temperature=0.2,
            max_output_tokens=1000,
            top_p=0.9
        )
        
        line_dict = {"drug_name": drug_name, "disease_name": disease_name, "answer": answer, "prompt": input_text}
        f_write.write(line_dict)