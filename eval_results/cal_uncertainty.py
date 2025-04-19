import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FOLDER_DIR = "./uncertainty_results"
OUTPUT_DIR = "./plots"
MODEL_NAME = "llama32_3b"
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

def create_calibration_plot(uncertainty_scores, accuracies, model_name):
    confidence_scores = [1 - u for u in uncertainty_scores]
    
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracies = np.zeros(len(bin_edges) - 1)
    bin_counts = np.zeros(len(bin_edges) - 1)
    
    for confidence, accuracy in zip(confidence_scores, accuracies):
        if confidence == 1.0:
            bin_idx = len(bin_edges) - 2
        else:
            bin_idx = int(confidence * 10)
        
        bin_accuracies[bin_idx] += accuracy
        bin_counts[bin_idx] += 1
    
    for i in range(len(bin_accuracies)):
        if bin_counts[i] > 0:
            bin_accuracies[i] /= bin_counts[i]
    
    perfect_calibration = bin_centers
    error = np.sum(np.abs(bin_accuracies - perfect_calibration) * bin_counts) / np.sum(bin_counts)
    error = round(error * 100, 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.bar(bin_centers, bin_accuracies, width=0.08, color='blue', edgecolor='black', label='Outputs')
    
    for i, (center, acc) in enumerate(zip(bin_centers, bin_accuracies)):
        if acc < center:
            ax.bar(center, center - acc, bottom=acc, width=0.08, color='lightpink', 
                  edgecolor='red', hatch='///', alpha=0.7, label='Gap' if i==0 else "")
        elif acc > center:
            ax.bar(center, acc - center, bottom=center, width=0.08, color='lightpink',
                  edgecolor='red', hatch='///', alpha=0.7, label='Gap' if i==0 else "")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, color='gray')
    
    ax.text(0.5, 0.2, f"Error={error}", bbox=dict(facecolor='lightgray', alpha=0.8), fontsize=12)
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    save_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return error

all_uncertainties = []
all_accuracies = []
file_results = {}

for filename in os.listdir(FILE_FOLDER):
    file_path = os.path.join(FILE_FOLDER, filename)
    if os.path.isfile(file_path):
        file_uncertainties = []
        file_accuracies = []
        incorrect_uncertainty_score = 0
        
        f_read = open(file_path, 'r', encoding='utf-8')
        for line in f_read:
            line = json.loads(line)
            answer_lst = line['answer']
            ground_truth = line['label']
            clean_answer_lst = []

            for answer in answer_lst:
                clean_answer = answer_extractor(answer)
                clean_answer_lst.append(clean_answer)
            
            uncertainty_score = cal_uncertainty(clean_answer_lst)
            acc_score = cal_acc(clean_answer_lst, ground_truth)
            
            file_uncertainties.append(uncertainty_score)
            file_accuracies.append(acc_score)
            all_uncertainties.append(uncertainty_score)
            all_accuracies.append(acc_score)

            if acc_score == 0:
                if incorrect_uncertainty_score == 0:
                    incorrect_uncertainty_score = uncertainty_score
                else:
                    incorrect_uncertainty_score = (uncertainty_score + incorrect_uncertainty_score) / 2
        
        file_results[filename] = {
            'accuracy': np.mean(np.array(file_accuracies)),
            'uncertainties': file_uncertainties
        }
        
        
        file_model_name = f"{os.path.splitext(filename)[0]}"
        file_error = create_calibration_plot(file_uncertainties, file_accuracies, file_model_name)
        print(f"File: {filename}, Calibration Error: {file_error}")

# error = create_calibration_plot(all_uncertainties, all_accuracies, MODEL_NAME)
# print(f"Overall Calibration Error: {error}")