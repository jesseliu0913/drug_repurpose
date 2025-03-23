import os
import json
import jsonlines
import numpy as np


file_folder = "./results/llama31_8b"
answer_dict = {"contraindication":"no", "indication":"yes"}

for filename in os.listdir(file_folder):
    file_path = os.path.join(file_folder, filename)
    tp = fp = tn = fn = 0
    total_count = 0
    correct = 0

    with open(file_path, 'r') as f_read:
        for line in f_read:
            line_dict = {}
            yes_answer = []
            no_answer = []

            info_dict = json.loads(line)
            for drug, answer in zip(info_dict['drug_name'], info_dict['answer']):
                total_count += 1
                label = answer_dict.get(drug[-1])  
                pred = answer.split("\n")[0].split(".")[0].lower().strip()  

                if label == pred:
                    correct += 1
                
                if label == "yes":
                    if pred == "yes":
                        tp += 1
                    else:
                        fn += 1
                elif label == "no":
                    if pred == "no":
                        tn += 1
                    else:
                        fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0               

        print(filename)   
        print("Acc:", correct / total_count)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        # f_write.write(line_dict)



