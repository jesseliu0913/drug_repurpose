import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter


FILE_FOLDER = "/playpen/jesse/drug_repurpose/k_path/eval_results"
model_folders = [f for f in os.listdir(FILE_FOLDER) 
              if os.path.isdir(os.path.join(FILE_FOLDER, f))]

def extract_answer(text: str) -> str | None:
    pattern = re.compile(r'Answer:\s*([^.\n]*)')
    m = pattern.search(text)
    return m.group(1).strip() if m else None

def calculate_score(label: str, answer: str, score_type: str) -> float:
    if score_type == 'binary':
        return 1.0 if label in answer else 0.0
    elif score_type == 'generation':
        ref_tokens = label.split()
        pred_tokens = answer.split()

        if not ref_tokens or not pred_tokens:
            return 0.0
        
        common = sum((Counter(ref_tokens) & Counter(pred_tokens)).values())
        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)

        return 2 * precision * recall / (precision + recall)
    else:
        raise ValueError("Unknown score type: {}".format(score_type))

for model_folder in model_folders:
    model_path = os.path.join(FILE_FOLDER, model_folder)
    print("Processing model folder:", model_folder)

    files = [file for file in os.listdir(model_path)
             if os.path.isfile(os.path.join(model_path, file))]
    
    for file in files:
        file_score = []

        file_name = file.split(".")[0]
        print(f"File in processing: {file_name}")
        file_path = os.path.join(model_path, file)
        file_data = pd.read_json(file_path, lines=True)

        if file_name in ['ddinter', 'pharmaDB']:
            score_type = 'binary'
        else:
            score_type = 'generation'

        for i in tqdm(range(len(file_data))):
            row = file_data.iloc[i]
            label = row['label']
            answer = extract_answer(row['answer']) 
            score = calculate_score(label, answer, score_type) if answer else 0.0
            file_score.append(score)
        
        print(f"File {file_name} score: {np.mean(file_score)}")


# python cal_score.py
