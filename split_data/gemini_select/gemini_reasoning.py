import os
import ast
import time
import json
import random
import argparse
import jsonlines
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions


def call_gemini(message, api_keys, temperature=0.7, max_output_tokens=1000, top_p=0.9, max_retries=10, initial_delay=2):
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


if __name__ == "__main__":
    api_keys = [
        "AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI",
        "AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc",
        "AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I",
        "AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw",
        "AIzaSyAsmMUeXmkOKwjmx__-rhZhyCevd5gllFc"
    ]
    
    file_path = f"/playpen/jesse/drug_repurpose/split_data/data/train.jsonl"
    f_write = jsonlines.open("./gemini_reasoning/reasoning_train.jsonl", "a")
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            prompt = line['prompt']
            
            prefix = f"{prompt}\nPlease provide the reasoning process to support the logical relationship described above in a single paragraph. REASON:"
            response = call_gemini(
                message=prefix,
                api_keys=api_keys,
                temperature=0.7,
                max_output_tokens=1000
            )
            
            line['reasoning'] = response
            f_write.write(line)