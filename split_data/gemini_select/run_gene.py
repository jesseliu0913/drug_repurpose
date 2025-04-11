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
parser.add_argument("--start_idx", type=int, help="The start index for data select")
parser.add_argument("--end_idx", type=int, help="The end index for data select")
parser.add_argument("--key_idx", type=int, help="The gemini key index choosed")
args = parser.parse_args()


gene_data = "/playpen/jesse/drug_repurpose/split_data/data/ddgene.jsonl"
# gene_data = "/playpen/jesse/drug_repurpose/split_data/data/dd_gene_negative.jsonl"

if args.key_idx == 0:
    user_key = "AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI"
elif args.key_idx == 1:
    user_key = "AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc"
elif args.key_idx == 2:
    user_key = "AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I"
elif args.key_idx == 3:
    user_key = "AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw"

genai.configure(api_key=user_key)
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

file_path = f"./output/positive/gene_{args.start_idx}_{args.end_idx}.jsonl"
f_write = jsonlines.open(file_path, "a")
with open(gene_data, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = json.loads(line)
        if idx >= args.start_idx and idx < args.end_idx:
            drug_name = line['drug_name']
            disease_name = line['disease_name']
            prompt = f"Is {disease_name} an indication for {drug_name}?"
            input_text = f"Question: {prompt} directly answer me with $YES$ or $NO$\nANSWER:"

            answer = call_gemini(
                message=input_text,
                temperature=0.2,
                max_output_tokens=2,
                top_p=0.9
            )
            
            line_dict = {"drug_name": drug_name, "disease_name": disease_name, "answer": answer, "prompt": input_text}
            f_write.write(line_dict)
        else:
            continue

# nohup python run_gene.py --start_idx 0 --end_idx 1000 --key_idx 0 > g_1.lpg 2>&1 &
# nohup python run_gene.py --start_idx 1000 --end_idx 2000 --key_idx 1 > g_2.lpg 2>&1 &