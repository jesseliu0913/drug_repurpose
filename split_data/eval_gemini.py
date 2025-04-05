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
Question: Is Metformin an indication for stroke disorder?
REASONING: Metformin is an antidiabetic medication primarily indicated for the treatment of type 2 diabetes mellitus, not for stroke or stroke disorders. While some studies suggest metformin may have secondary vascular benefits (like reducing the risk of cardiovascular events), it is not indicated or approved for the treatment or prevention of stroke itself. Therefore, its use in stroke disorders would be considered off-label and not based on formal clinical guidelines.
ANSWER:$NO$
Question: Is Ceftizoxime an indication for infectious meningitis?
REASONING: Ceftizoxime is an indication for infectious meningitis because it is a third-generation cephalosporin with broad-spectrum activity against key meningitis pathogens such as Neisseria meningitidis, Haemophilus influenzae, and some Streptococcus pneumoniae strains. It effectively penetrates the blood-brain barrier and reaches therapeutic levels in the cerebrospinal fluid, making it suitable for CNS infections like meningitis, particularly when the causative organism is susceptible.
ANSWER:$YES$
"""

raw_shot = """
Question: Is Metformin an indication for stroke disorder?
ANSWER:$NO$
Question: Is Ceftizoxime an indication for infectious meningitis?
ANSWER:$YES$
"""

genai.configure(api_key="AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI")
# AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI
# AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc
# AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I
# AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw

def call_gemini(message, temperature=0.7, max_output_tokens=1000, top_p=0.9, max_retries=10, initial_delay=2):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    for attempt in range(max_retries + 1):
        try:
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
            if attempt == max_retries:
                raise
                
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

os.makedirs(args.output_path, exist_ok=True)
prompt_type = args.prompt_type
file_path = f"{args.output_path}/{prompt_type}.jsonl"

test_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/test_data.csv")
node_data = pd.read_csv("/playpen/jesse/drug_repurpose/PrimeKG/nodes.csv")
start_index = 463
with jsonlines.open(file_path, "a") as f_write:
    for index, row in test_data.iloc[start_index:].iterrows():
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
        
        phenotype = phenotype[:10]
        gene = gene[:10]

        if prompt_type == "phenotype":
            prefix = f"{disease_name} includes the following phenotypes: {phenotype}"
            question = f"{prefix}\nIs {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} directly answer me with $YES$ or $NO$\nANSWER:"
        elif prompt_type == "cot":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {question} let's think step by step and then answer me with $YES$ or $NO$\nREASONING:\nANSWER:"
        elif prompt_type == "fcot":  
            question = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"{two_shot}\nQuestion: {question}."
        elif prompt_type == "gene":  
            prefix = f"{disease_name} includes the following genes: {gene}"
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
            temperature=0.2,
            max_output_tokens=1000,
            top_p=0.9
        )
        
        line_dict = {"drug_name": drug_name, "disease_name": disease_name, "answer": answer, "prompt": input_text}
        f_write.write(line_dict)