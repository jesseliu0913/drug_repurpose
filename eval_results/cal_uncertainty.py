import os
import json
import math
import numpy as np
import pandas as pd


FOLDER_DIR = "./uncertainty_results"
MODEL_NAME = "llama32_3b_loracot"
FILE_FOLDER = os.path.join(FOLDER_DIR, MODEL_NAME)

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


def cal_uncertainty(answer_lst):
    n = len(answer_lst)
    if n == 0:
        return 0.0
    count_0 = answer_lst.count(0)
    count_1 = answer_lst.count(1)
    p_0 = count_0 / n
    p_1 = count_1 / n
    entropy = 0
    if p_0 > 0:
        entropy -= p_0 * math.log2(p_0)
    if p_1 > 0:
        entropy -= p_1 * math.log2(p_1)
    return entropy

def answer_extractor(answer):
    answer = answer.split("\n")[0].lower()
    if "no" in answer:
        answer = 0
    elif "yes" in answer:
        answer = 1
    return answer


for filename in os.listdir(FILE_FOLDER):
    file_path = os.path.join(FILE_FOLDER, filename)
    if os.path.isfile(file_path):
        total_acc_lst = []
        total_uncertainty_score = 0
        f_read = open(file_path, 'r', encoding='utf-8')
        for line in f_read:
            line = json.loads(line)
            answer_lst = line['answer']
            ground_truth = line['label']
            clean_answer_lst = []

            for answer in answer_lst:
                clean_answer = answer_extractor(answer)
                clean_answer_lst.append(clean_answer)
            
            # Calculate uncertainty
            uncertainty_score = cal_uncertainty(clean_answer_lst)
            acc_score = cal_acc(clean_answer_lst, ground_truth)
            total_acc_lst.append(acc_score)

            if acc_score == 0:
                total_uncertainty_score  = (uncertainty_score + total_uncertainty_score) / 2
        print(filename)
        print(np.mean(np.array(total_acc_lst)))
        print(total_uncertainty_score)
