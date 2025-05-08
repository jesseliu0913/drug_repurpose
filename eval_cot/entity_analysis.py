import os
import json
import argparse
import google.generativeai as genai
from tqdm import tqdm

def call_gemini(api_key, question, answer):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Extract all relevant knowledge triples from the following answer to a biomedical question about whether a drug is indicated for a disease.
    
    Question: {question}
    
    Answer: {answer}
    
    Focus on extracting triples that provide evidence for or against the indication relationship, such as:
    - Drug affects Gene/Protein/Pathway
    - Disease is associated with Gene/Protein/Pathway
    - Drug has mechanism/action
    - Disease has mechanism/pathophysiology
    - Disease has exposure to Drug
    - Disease has phenotype
    - Drug is used for Disease
    - Drug shows efficacy against Disease
    - Drug has adverse effects relevant to Disease
    - Drug interacts with other drugs used for Disease
    
    Format your response as a JSON object with the following structure:
    {{
        "drug_related_triples": [
            {{"subject": string, "relationship": string, "object": string}}
        ],
        "disease_related_triples": [
            {{"subject": string, "relationship": string, "object": string}}
        ],
        "drug_disease_triples": [
            {{"subject": string, "relationship": string, "object": string}}
        ]
    }}
    
    Extract as many relevant triples as possible, but ensure each triple is clearly supported by the text.
    """
    
    response = model.generate_content(prompt)
    try:
        response_text = response.text
        results = json.loads(response_text)
        return results
    except json.JSONDecodeError:
        return {"error": "Failed to parse Gemini response as JSON", "raw_response": response.text}

def extract_triples(api_key, data_path, output_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    processed_count = 0
    existing_results = []
    
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                existing_results.append(json.loads(line))
        processed_count = len(existing_results)
        print(f"Found {processed_count} already processed items in {output_path}")
    
    results = []
    with open(output_path, 'a') as f:
        for item in tqdm(data[processed_count:], initial=processed_count, total=len(data)):
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            extracted_triples = call_gemini(api_key, question, answer)
            
            result = {
                "question": question,
                "answer": answer,
                "extracted_triples": extracted_triples
            }
            
            # results.append(result)
            f.write(json.dumps(result) + '\n')
            f.flush()
    
    remaining_count = len(data) - processed_count
    print(f"Processed {remaining_count} additional items. Total items in {output_path}: {processed_count + remaining_count}")


def main():
    parser = argparse.ArgumentParser(description="Resume extraction of biomedical knowledge triples using Gemini")
    parser.add_argument('--api_key', required=True, help='Gemini API key')
    parser.add_argument('--file_types', nargs='+', required=True, help='List of file types to evaluate (e.g., llama32_1b llama32_3b)')
    parser.add_argument('--results_dir', default='../eval_cot/results', help='Base directory containing result files')

    args = parser.parse_args()
    OUTPUT_FOLDER = "./results"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for file_type in args.file_types:
        os.makedirs(os.path.join(OUTPUT_FOLDER, file_type), exist_ok=True)
    
    for file_type in args.file_types:
        input_dir = "../eval_results/results"
        file_folder = os.path.join(input_dir, file_type)
        output_folder = os.path.join(OUTPUT_FOLDER, file_type)
        input_file_path = os.path.join(file_folder, 'cot.jsonl')
        output_file_path = os.path.join(output_folder, 'extracted_triplets.jsonl')

        print(input_file_path)
        extract_triples(args.api_key, input_file_path, output_file_path)

if __name__ == "__main__":
    main()