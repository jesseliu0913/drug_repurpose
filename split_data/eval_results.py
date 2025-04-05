import os
import re
import glob
import argparse
import pandas as pd
import jsonlines
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import nltk
from nltk.tokenize import word_tokenize

def extract_yes_no(answer):
    if "$YES$" in answer:
        return "YES"
    elif "$NO$" in answer:
        return "NO"
    
    answer = answer.upper()
    if re.search(r'\bYES\b', answer):
        return "YES"
    elif re.search(r'\bNO\b', answer):
        return "NO"
    
    return None

def calculate_token_variance(answers):
    token_counts = []
    for answer in answers:
        if answer:
            tokens = word_tokenize(answer)
            token_counts.append(len(tokens))
    
    if token_counts:
        return np.var(token_counts)
    return 0

def calculate_metrics(jsonl_file, ground_truth_file=None):
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        truth_df = pd.read_csv(ground_truth_file)
        for _, row in truth_df.iterrows():
            key = (row['drug_name'], row['disease_name'])
            ground_truth[key] = 1 if row['relation'] == 'positive' else 0
    
    y_true = []
    y_pred = []
    records = []
    all_answers = []
    
    try:
        with jsonlines.open(jsonl_file, 'r') as reader:
            for obj in reader:
                drug = obj.get('drug_name')
                disease = obj.get('disease_name')
                answer = obj.get('answer', '')
                all_answers.append(answer)
                prediction_text = extract_yes_no(answer)
                
                prediction = 1 if prediction_text == "YES" else 0
                
                if (drug, disease) in ground_truth and prediction_text is not None:
                    true_label = ground_truth[(drug, disease)]
                    y_true.append(true_label)
                    y_pred.append(prediction)
                    
                records.append({
                    'drug_name': drug,
                    'disease_name': disease,
                    'predicted_answer': prediction_text,
                    'ground_truth': "YES" if (drug, disease) in ground_truth and ground_truth[(drug, disease)] == 1 else "NO",
                    'correct': prediction == ground_truth.get((drug, disease)) if (drug, disease) in ground_truth else None
                })
    except Exception as e:
        print(f"Error reading file {jsonl_file}: {e}")
        return {}, pd.DataFrame()
    
    metrics = {}
    if len(y_true) > 0 and len(y_pred) > 0:
        positive_samples = sum(y_true)
        if positive_samples > 0 and positive_samples < len(y_true):
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = 0
            metrics['precision'] = 0
            metrics['recall'] = 0
    
    metrics['token_variance'] = calculate_token_variance(all_answers)
    
    return metrics, pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from model predictions")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing result folders")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth CSV file")
    parser.add_argument("--output_file", type=str, default="metrics_results.csv", help="Output CSV file")
    args = parser.parse_args()
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    all_files = glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    results = []
    
    if args.ground_truth and os.path.exists(args.ground_truth):
        truth_df = pd.read_csv(args.ground_truth)
        positive_count = sum(truth_df['relation'] == 'positive')
        total_count = len(truth_df)
        print(f"Ground truth distribution: {positive_count}/{total_count} positive examples ({positive_count/total_count:.2%})")
    
    for file_path in all_files:
        prompt_type = os.path.basename(file_path).replace('.jsonl', '')
        model_name = os.path.basename(os.path.dirname(file_path))
        
        print(f"Processing: {model_name}/{prompt_type}")
        metrics, df = calculate_metrics(file_path, args.ground_truth)
        
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not df.empty and 'predicted_answer' in df.columns:
            yes_count = sum(1 for p in df['predicted_answer'] if p == "YES")
            no_count = sum(1 for p in df['predicted_answer'] if p == "NO")
            valid_count = sum(1 for p in df['predicted_answer'] if p is not None)
        else:
            yes_count = no_count = valid_count = 0
        
        result = {
            'model': model_name,
            'prompt_type': prompt_type,
            'accuracy': metrics.get('accuracy', 0),
            'f1': metrics.get('f1', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'token_variance': metrics.get('token_variance', 0),
            'total': len(df),
            'yes_count': yes_count,
            'no_count': no_count,
            'yes_ratio': yes_count / valid_count if valid_count > 0 else 0,
            'valid_predictions': valid_count
        }
        
        results.append(result)
        print(f"Model: {model_name}, Prompt: {prompt_type}, Acc: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1', 0):.4f}, Token Var: {metrics.get('token_variance', 0):.2f}")
    
    result_df = pd.DataFrame(results)
    if args.output_file:
        result_df.to_csv(args.output_file, index=False)
    
    print("\nMETRICS SUMMARY:")
    for metric in ['accuracy', 'f1', 'token_variance']:
        print(f"\n{metric.upper()}:")
        pivot = pd.pivot_table(
            result_df, 
            values=metric,
            index='model',
            columns='prompt_type',
            aggfunc='mean'
        )
        print(pivot.round(4))
    
    print("\nPREDICTION DISTRIBUTION (YES %):")
    pivot = pd.pivot_table(
        result_df, 
        values='yes_ratio',
        index='model',
        columns='prompt_type',
        aggfunc='mean'
    )
    print((pivot * 100).round(1))

if __name__ == "__main__":
    main()

# python eval_results.py  --input_dir ./results --ground_truth ./test_data.csv