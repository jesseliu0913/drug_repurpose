import json
import time
import random
import google.generativeai as genai
from google.api_core import exceptions
from tqdm import tqdm

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

def process_jsonlines_file(input_file, output_file):
    prompt_template = """
    Provide clear, concise reasoning for the following prompt:
    
    {prompt}
    
    Give a step-by-step logical explanation that is accurate and to the point.
    """
    with open(input_file, 'r', encoding='utf-8') as in_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:
        
        lines = in_file.readlines()
        for line in tqdm(lines, desc="Processing prompts"):
            data = json.loads(line)
            
            prompt = data.get("prompt", "")
            
            if prompt:
                gemini_prompt = prompt_template.format(prompt=prompt)
                try:
                    reasoning = call_gemini(gemini_prompt, temperature=0.3)  
                    data["reasoning"] = reasoning
                    
                except Exception as e:
                    print(f"Error processing prompt: {prompt[:50]}...")
                    print(f"Error details: {e}")
                    data["reasoning"] = f"None"
            
            out_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            time.sleep(0.5)

if __name__ == "__main__":
    input_jsonl_file = "./data/combined.jsonl"
    output_jsonl_file = "./data/gemini_reasoning.jsonl"
    
    genai.configure(api_key="AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw")
    
    process_jsonlines_file(input_jsonl_file, output_jsonl_file)
    print(f"Processing complete. Results saved to {output_jsonl_file}")