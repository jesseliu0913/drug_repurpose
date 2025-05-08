import os
import json
import argparse
import re
from collections import defaultdict

def extract_json_from_response(raw_response):
    """Extract JSON from responses that contain markdown code blocks."""
    if not raw_response:
        return None
    
    # First check for JSON wrapped in markdown
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try direct JSON parsing
    try:
        return json.loads(raw_response)
    except:
        pass
    
    # If all else fails, try to extract individual scores
    scores = {}
    patterns = {
        'logical_analysis': r'"logical_analysis":\s*{\s*"score":\s*(\d+)',
        'biological_information': r'"biological_information":\s*{\s*"score":\s*(\d+)',
        'evidence_integration': r'"evidence_integration":\s*{\s*"score":\s*(\d+)',
        'overall_diversity': r'"overall_diversity":\s*{\s*"score":\s*(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, raw_response)
        if match:
            scores[key] = {'score': int(match.group(1))}
    
    return scores if scores else None

def analyze_diversity_scores(file_path):
    """Analyze diversity evaluation scores from a JSONL file."""
    score_stats = {
        'logical_analysis': [],
        'biological_information': [],
        'evidence_integration': [],
        'overall_diversity': [],
        'total_evaluations': 0,
        'successful_evaluations': 0,
        'failed_evaluations': 0
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                score_stats['total_evaluations'] += 1
                
                # Get evaluation data
                evaluation = item.get('diversity_evaluation', {})
                
                # Handle parsing errors
                if isinstance(evaluation, dict) and 'error' in evaluation:
                    raw_response = evaluation.get('raw_response', '')
                    evaluation = extract_json_from_response(raw_response)
                    if not evaluation:
                        score_stats['failed_evaluations'] += 1
                        continue
                
                # Extract scores
                extracted_scores = False
                for category in ['logical_analysis', 'biological_information', 
                               'evidence_integration', 'overall_diversity']:
                    if category in evaluation and 'score' in evaluation[category]:
                        score_stats[category].append(evaluation[category]['score'])
                        extracted_scores = True
                
                if extracted_scores:
                    score_stats['successful_evaluations'] += 1
                else:
                    score_stats['failed_evaluations'] += 1
                    
            except:
                score_stats['failed_evaluations'] += 1
                continue
    
    return score_stats

def calculate_statistics(scores):
    """Calculate statistics for a list of scores."""
    if not scores:
        return {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
    
    return {
        'mean': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores),
        'count': len(scores)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze diversity evaluation scores")
    parser.add_argument('--results_dir', default='./results', help='Directory containing result files')
    parser.add_argument('--file_types', nargs='+', required=True, help='List of file types to analyze')
    
    args = parser.parse_args()
    
    print("\n=== DIVERSITY SCORE ANALYSIS ===\n")
    
    all_stats = {}
    
    for file_type in args.file_types:
        file_path = os.path.join(args.results_dir, file_type, 'llm_judge.jsonl')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        stats = analyze_diversity_scores(file_path)
        all_stats[file_type] = stats
        
        print(f"Model: {file_type}")
        print(f"  Total evaluations: {stats['total_evaluations']}")
        print(f"  Successful evaluations: {stats['successful_evaluations']}")
        print(f"  Failed evaluations: {stats['failed_evaluations']}")
        
        if stats['successful_evaluations'] > 0:
            success_rate = (stats['successful_evaluations'] / stats['total_evaluations']) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        print("\n  Score Statistics:")
        for category in ['logical_analysis', 'biological_information', 
                        'evidence_integration', 'overall_diversity']:
            cat_stats = calculate_statistics(stats[category])
            if cat_stats['count'] > 0:
                print(f"    {category}:")
                print(f"      Mean: {cat_stats['mean']:.2f}")
                print(f"      Min: {cat_stats['min']}")
                print(f"      Max: {cat_stats['max']}")
                print(f"      Count: {cat_stats['count']}")
        
        print()
    
    # Summary comparison
    if len(all_stats) > 1:
        print("=== COMPARISON SUMMARY ===")
        print(f"{'Model':<20} {'Logical':<10} {'Biological':<12} {'Evidence':<10} {'Overall':<10} {'Success':<10}")
        print("-" * 75)
        
        for file_type, stats in all_stats.items():
            logical_mean = sum(stats['logical_analysis']) / len(stats['logical_analysis']) if stats['logical_analysis'] else 0
            bio_mean = sum(stats['biological_information']) / len(stats['biological_information']) if stats['biological_information'] else 0
            evidence_mean = sum(stats['evidence_integration']) / len(stats['evidence_integration']) if stats['evidence_integration'] else 0
            overall_mean = sum(stats['overall_diversity']) / len(stats['overall_diversity']) if stats['overall_diversity'] else 0
            success_rate = (stats['successful_evaluations'] / stats['total_evaluations'] * 100) if stats['total_evaluations'] > 0 else 0
            
            print(f"{file_type:<20} {logical_mean:<10.2f} {bio_mean:<12.2f} {evidence_mean:<10.2f} {overall_mean:<10.2f} {success_rate:<9.1f}%")

if __name__ == "__main__":
    main()


# python eval_llm_data.py --file_types llama32_1b llama32_1b_loracot llama32_3b llama32_3b_loracot --results_dir ./results