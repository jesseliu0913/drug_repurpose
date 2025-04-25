import os
import json
import pandas as pd
import numpy as np


file_path = "/playpen/jesse/drug_repurpose/eval_results/uncertainty_results/gemini/cot.jsonl"
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        data.append(json_obj)

def answer_extractor_cot(answer):
    if "$NO$" in answer.upper():
        answer = 0
    elif "$YES$" in answer.upper():
        answer = 1
    elif "YES" in answer.upper():
        answer = 1
    elif "$NO$" in answer.upper():
        answer = 0
    return answer

def cal_acc(clean_answer_lst, ground_truth):
    n = len(clean_answer_lst)
    if n == 0:
        return 0.0
    count_0 = clean_answer_lst.count(0)
    count_1 = clean_answer_lst.count(1)
    answer = "indication" if count_1 > count_0 else "contraindication"
    if answer == ground_truth:
        acc_score = 1
    else:
        acc_score = 0

    return acc_score

all_accuracies = []
for line in data:
    answer = line['answer']
    ground_truth = line['label']
    clean_answer_lst = []
    for an in answer:
        clean_answer = answer_extractor_cot(an)
        clean_answer_lst.append(clean_answer)
    acc_score = cal_acc(clean_answer_lst, ground_truth)
    all_accuracies.append(acc_score)

print(np.mean(np.array(all_accuracies)))

    