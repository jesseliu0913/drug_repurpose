import json
import os
import time
import random
import pyperclip
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

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_entity_info(item):
    """Extract the entity (gene or phenotype) information from an item"""
    # Try different possible keys for phenotype
    phenotype_keys = ['phenotype', 'phenotype_name', 'phenotype_name', 'disease']
    for key in phenotype_keys:
        if key in item and item[key]:
            return item[key], 'phenotype', key
    
    # Check if disease_name contains keywords suggesting it's a phenotype
    if 'disease_name' in item:
        disease = item['disease_name'].lower()
        if 'disease' in disease or 'syndrome' in disease or 'disorder' in disease:
            return item['disease_name'], 'phenotype', 'disease_name'
    
    # Try gene keys
    gene_keys = ['gene_name', 'gene']
    for key in gene_keys:
        if key in item and item[key]:
            return item[key], 'gene', key
    
    # Default fallback: check all keys for gene or phenotype mentions
    for key in item.keys():
        if 'gene' in key.lower() and item[key]:
            return item[key], 'gene', key
        elif ('pheno' in key.lower() or 'disease' in key.lower()) and item[key]:
            return item[key], 'phenotype', key
    
    # Final fallback
    return "Unknown Entity", "entity", None

def get_processed_items(file_path):
    processed = set()
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        drug = item.get('drug_name', '')
                        disease = item.get('disease_name', '')
                        
                        # Get entity info using the helper function
                        entity_name, _, _ = get_entity_info(item)
                        
                        # Use a combination of fields as a unique identifier
                        key = f"{drug}-{disease}-{entity_name}"
                        processed.add(key)
        return processed
    except Exception as e:
        print(f"Error reading processed items: {e}")
        return set()

def save_item(item, file_path, append=True):
    try:
        with open(file_path, 'a' if append else 'w', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Entry saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving item: {e}")
        return False

def generate_gemini_prompt(item):
    drug = item.get('drug_name', 'Unknown Drug')
    disease = item.get('disease_name', 'Unknown Disease')
    
    # Get entity info using the helper function
    entity_name, entity_type, _ = get_entity_info(item)
    
    answer = item.get('answer', '').strip().upper()
    
    # Extract YES/NO from the answer field
    if 'YES' in answer:
        indication_status = "indicated"
    elif 'NO' in answer:
        indication_status = "contraindicated"
    else:
        # Show the answer and let the user decide
        print(f"\nUnclear answer value: '{answer}'")
        indication_choice = input("Is this drug indicated (i) or contraindicated (c)? ").lower()
        indication_status = "indicated" if indication_choice == 'i' else "contraindicated"
    
    prompt = f"The drug {drug} is associated with {entity_name} ({entity_type}) and is {indication_status} for the disease {disease}. Please provide a clear, concise, and accurate one-paragraph explanation for this association and contraindication."
    
    return prompt

def interactive_editor(data, output_file, api_keys):
    print("\n===== INTERACTIVE REASONING EDITOR =====")
    print("Review each reasoning and choose to modify it or keep it as is.")
    print("When a reasoning is displayed, you can:")
    print("  - Enter 'k' or just press Enter to keep it as is")
    print("  - Enter 'm' to modify it")
    print("  - Enter 'g' to generate and use Gemini response")
    print("  - Enter 'c' to copy Gemini prompt to clipboard")
    print("  - Enter 'q' to save and quit\n")
    
    # Get already processed items
    processed_items = get_processed_items(output_file)
    print(f"Found {len(processed_items)} already processed items in {output_file}")
    
    total_items = len(data)
    processed_count = 0
    skipped_count = 0
    
    for i, item in enumerate(data):
        # Get entity information
        entity_name, entity_type, entity_key = get_entity_info(item)
        
        drug = item.get('drug_name', 'Unknown Drug')
        disease = item.get('disease_name', 'Unknown Disease')
        
        # Create a unique key for this item
        key = f"{drug}-{disease}-{entity_name}"
        
        # Skip if already processed
        if key in processed_items:
            skipped_count += 1
            continue
        
        reasoning = item.get('reasoning', '')
        answer = item.get('answer', '').strip()
        
        print(f"\n[Item {i+1}/{total_items} - Processed: {processed_count} - Skipped: {skipped_count}]")
        print(f"Drug: {drug}")
        print(f"Disease: {disease}")
        print(f"{entity_type.capitalize()}: {entity_name}")
        print(f"Answer: {answer}")
        print("\nReasoning:")
        print("-" * 80)
        print(reasoning)
        print("-" * 80)
        
        while True:
            choice = input("\nKeep (k/Enter), Modify (m), Generate Gemini response (g), Copy prompt (c), or Quit (q)? ").lower()
            
            if choice == 'q':
                # Don't automatically save remaining items
                print(f"\nQuitting. Processed {processed_count} items out of {total_items} (skipped {skipped_count} already processed items).")
                return
            elif choice == 'c':
                prompt = generate_gemini_prompt(item)
                try:
                    pyperclip.copy(prompt)
                    print("\nGemini prompt copied to clipboard:")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)
                except Exception as e:
                    print(f"\nCouldn't copy to clipboard: {e}")
                    print("\nGemini prompt (please copy manually):")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)
            elif choice == 'g':
                prompt = generate_gemini_prompt(item)
                print("\nPrompt to be sent to Gemini:")
                print("-" * 80)
                print(prompt)
                print("-" * 80)
                
                confirm = input("\nSend this prompt to Gemini? (y/n): ").lower()
                if confirm != 'y':
                    print("Gemini request canceled.")
                    continue
                
                print("\nGenerating Gemini response...")
                
                try:
                    gemini_response = call_gemini(prompt, api_keys)
                    print("\nGemini Response:")
                    print("-" * 80)
                    print(gemini_response)
                    print("-" * 80)
                    
                    use_response = input("\nUse this Gemini response? (y/n): ").lower()
                    if use_response == 'y':
                        modified_item = item.copy()
                        modified_item['reasoning'] = gemini_response
                        save_item(modified_item, output_file)
                        processed_count += 1
                        processed_items.add(key)
                        break
                except Exception as e:
                    print(f"\nError getting Gemini response: {e}")
            elif choice == 'm':
                print("\nEnter your modified reasoning (press Enter twice when finished):")
                
                lines = []
                while True:
                    line = input()
                    if line == "" and (not lines or lines[-1] == ""):
                        break
                    lines.append(line)
                
                if lines and lines[-1] == "":
                    lines.pop()
                
                modified_reasoning = "\n".join(lines)
                
                modified_item = item.copy()
                modified_item['reasoning'] = modified_reasoning
                save_item(modified_item, output_file)
                processed_count += 1
                processed_items.add(key)
                break
            elif choice == 'k' or choice == '':
                save_item(item, output_file)
                processed_count += 1
                processed_items.add(key)
                break
            else:
                print("Invalid option. Please try again.")

def main():
    api_keys = [
        "AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI",
        "AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc",
        "AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I", 
        "AIzaSyAmGjvNInLdFV7N9Oxp3FhJFIE81WUdDgw",
        "AIzaSyAsmMUeXmkOKwjmx__-rhZhyCevd5gllFc"
    ]
    
    input_file = "./reasoning_train.jsonl"
    
    data = load_jsonl(input_file)
    if not data:
        print("Could not load data from the file.")
        return
    
    print(f"Loaded {len(data)} items from {input_file}")
    
    output_file = "./modified_reasoning.jsonl"
    interactive_editor(data, output_file, api_keys)
    
    print(f"\nAll data has been processed and saved to {output_file}")

if __name__ == "__main__":
    main()