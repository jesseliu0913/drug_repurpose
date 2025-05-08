import os
import json
import argparse
import re
from collections import defaultdict

def extract_json_from_response(raw_response):
    """Extract JSON from responses that contain markdown code blocks."""
    if not raw_response:
        return None
    
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    try:
        return json.loads(raw_response)
    except:
        return None

def count_triples(file_path):
    """Count all triples in a JSONL file."""
    triple_counts = {
        'drug_related_triples': 0,
        'disease_related_triples': 0,
        'drug_disease_triples': 0,
        'total_triples': 0,
        'examples_processed': 0,
        'examples_with_triples': 0
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                triple_counts['examples_processed'] += 1
                
                # Get triples
                triples = item.get('extracted_triples', {})
                
                # Handle parsing errors
                if isinstance(triples, dict) and 'error' in triples:
                    raw_response = triples.get('raw_response', '')
                    triples = extract_json_from_response(raw_response)
                    if not triples:
                        continue
                
                example_has_triples = False
                
                # Count triples by type
                for triple_type in ['drug_related_triples', 'disease_related_triples', 'drug_disease_triples']:
                    if triple_type in triples and triples[triple_type]:
                        count = len(triples[triple_type])
                        triple_counts[triple_type] += count
                        triple_counts['total_triples'] += count
                        example_has_triples = True
                
                if example_has_triples:
                    triple_counts['examples_with_triples'] += 1
                    
            except:
                continue
    
    return triple_counts

def main():
    parser = argparse.ArgumentParser(description="Count triples extracted by each model")
    parser.add_argument('--results_dir', default='./results', help='Directory containing result files')
    parser.add_argument('--file_types', nargs='+', required=True, help='List of file types to analyze')
    
    args = parser.parse_args()
    
    print("\n=== TRIPLE COUNT ANALYSIS ===\n")
    
    all_counts = {}
    
    for file_type in args.file_types:
        file_path = os.path.join(args.results_dir, file_type, 'extracted_triplets.jsonl')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        counts = count_triples(file_path)
        all_counts[file_type] = counts
        
        print(f"Model: {file_type}")
        print(f"  Total examples processed: {counts['examples_processed']}")
        print(f"  Examples with triples: {counts['examples_with_triples']}")
        print(f"  Drug-related triples: {counts['drug_related_triples']}")
        print(f"  Disease-related triples: {counts['disease_related_triples']}")
        print(f"  Drug-disease triples: {counts['drug_disease_triples']}")
        print(f"  TOTAL TRIPLES: {counts['total_triples']}")
        
        if counts['examples_processed'] > 0:
            avg_per_example = counts['total_triples'] / counts['examples_processed']
            print(f"  Average triples per example: {avg_per_example:.2f}")
            
            success_rate = (counts['examples_with_triples'] / counts['examples_processed']) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        print()
    
    # Summary comparison
    if len(all_counts) > 1:
        print("=== COMPARISON SUMMARY ===")
        print(f"{'Model':<15} {'Total Triples':<15} {'Avg/Example':<15} {'Success Rate':<15}")
        print("-" * 60)
        
        for file_type, counts in all_counts.items():
            total = counts['total_triples']
            avg = total / counts['examples_processed'] if counts['examples_processed'] > 0 else 0
            success = (counts['examples_with_triples'] / counts['examples_processed'] * 100) if counts['examples_processed'] > 0 else 0
            print(f"{file_type:<15} {total:<15} {avg:<15.2f} {success:<14.1f}%")

if __name__ == "__main__":
    main()

# python eval_ea_data.py --file_types llama32_1b llama32_1b_loracot llama32_3b llama32_3b_loracot --results_dir ./results