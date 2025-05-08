import os
import json
import argparse
import google.generativeai as genai
from tqdm import tqdm

def call_gemini(api_key, question, answer):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    Evaluate the following answer to a biomedical question for its diversity and depth of analysis.

    Question: {question}

    Answer: {answer}

    Please assess:
    1. Logical Analysis: Does the answer provide multiple logical pathways or perspectives?
    2. Biological Information: Does the answer incorporate diverse biological information (genes, pathways, mechanisms, etc.)?
    3. Evidence Integration: Does the answer connect multiple sources or types of evidence?

    Rate each aspect on a scale of 1-5, where 1 is poor and 5 is excellent.
    Provide specific examples from the answer for each rating.

    Finally, give an overall diversity score from 1-10 and a brief justification.

    Format your response as a JSON object with the following structure:
    {{
        "logical_analysis": {{
            "score": int,
            "examples": list of strings
        }},
        "biological_information": {{
            "score": int,
            "examples": list of strings
        }},
        "evidence_integration": {{
            "score": int,
            "examples": list of strings
        }},
        "overall_diversity": {{
            "score": int,
            "justification": string
        }}
    }}
    """

    response = model.generate_content(prompt)
    try:
        response_text = response.text
        results = json.loads(response_text)
        return results
    except json.JSONDecodeError:
        return {"error": "Failed to parse Gemini response as JSON", "raw_response": response.text}

def evaluate_diversity(api_key, data_path, output_path):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    processed_count = 0
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for processed_count, _ in enumerate(f, start=1):
                pass
        print(f"Found {processed_count} already processed items in {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'w').close()
        print(f"Created new output file at {output_path}")

    with open(output_path, 'a') as fout:
        for item in tqdm(data[processed_count:], initial=processed_count, total=len(data), desc="Evaluating diversity"):  
            question = item.get("question", "")
            answer = item.get("answer", "")
            evaluation = call_gemini(api_key, question, answer)

            result = {
                "question": question,
                "answer": answer,
                "diversity_evaluation": evaluation
            }
            fout.write(json.dumps(result) + '\n')
            fout.flush()

    total = len(data)
    processed = processed_count + (total - processed_count)
    print(f"Processed {total - processed_count} new items. Total items in {output_path}: {processed}")

def main():
    parser = argparse.ArgumentParser(description="Resume evaluation of diversity of biomedical QA answers using Gemini")
    parser.add_argument('--api_key', required=True, help='Gemini API key')
    parser.add_argument('--file_types', nargs='+', required=True, help='List of file types to evaluate (e.g., llama32_1b llama32_3b)')
    parser.add_argument('--results_dir', default='../eval_results/results', help='Base directory containing result files')

    args = parser.parse_args()
    OUTPUT_FOLDER = "./results"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for file_type in args.file_types:
        os.makedirs(os.path.join(OUTPUT_FOLDER, file_type), exist_ok=True)
    
    for file_type in args.file_types:
        file_folder = os.path.join(args.results_dir, file_type)
        output_folder = os.path.join(OUTPUT_FOLDER, file_type)
        input_file_path = os.path.join(file_folder, 'cot.jsonl')
        output_file_path = os.path.join(output_folder, 'llm_judge.jsonl')

        evaluate_diversity(args.api_key, input_file_path, output_file_path)

if __name__ == "__main__":
    main()