import os
import json
import jsonlines
import numpy as np


file_path = "./results/llama32_1b/output_DD1k_10.jsonl"
# f_write = jsonlines.open("./results/llama32_1b/output_DD1k_10_clean", "a")
total_correct = []
with open(file_path, 'r') as f_read:
    for line in f_read:
        line_dict = {}
        yes_answer = []
        no_answer = []

        info_dict = json.loads(line)
        if info_dict['effect_type'] == "contraindication":
            groundtruth = "no"
        elif info_dict['effect_type'] == "indication":
            groundtruth = "yes"
        else:
            groundtruth = info_dict['effect_type']
        
        line_dict['drug_name'] = info_dict['drug_name']
        line_dict['target_name'] = info_dict['target_name']

        for answer in info_dict['answer']:
            clean_answer = answer.split("\n")[0].split(".")[0].lower()
            if "yes" in clean_answer:
                yes_answer.append(clean_answer)
            elif "no" in clean_answer:
                no_answer.append(clean_answer)
            else:
                print("invalid answer")

        if groundtruth == "no":
            true_rate = len(no_answer) / len(info_dict['answer']) if len(info_dict['answer']) != 0 else 0
            false_rate = 1-true_rate
        elif groundtruth == "yes":
            true_rate = len(yes_answer) / len(info_dict['answer']) if len(info_dict['answer']) != 0 else 0
            false_rate = 1-true_rate
        else:
            continue
        
        line_dict['effect_type'] = info_dict['effect_type']
        line_dict['groundtruth'] = groundtruth
        line_dict['true_rate'] = true_rate
        line_dict['false_rate'] = false_rate
        line_dict['answers'] = info_dict['answer']

        # f_write.write(line_dict)
        total_correct.append(true_rate)


print(np.mean(np.array(total_correct)))

