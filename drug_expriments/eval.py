import os
import json
import jsonlines
import numpy as np


file_path = "./results/llama32_1b/output_DD1k.jsonl"
total_correct = 0
correct = 0

with open(file_path, 'r') as f_read:
    for line in f_read:
        line_dict = {}
        yes_answer = []
        no_answer = []
        total_correct += 1

        info_dict = json.loads(line)
        if info_dict['effect_type'] == "contraindication":
            groundtruth = "no"
        elif info_dict['effect_type'] == "indication":
            groundtruth = "yes"
        else:
            groundtruth = info_dict['effect_type']
        
        line_dict['drug_name'] = info_dict['drug_name']
        line_dict['target_name'] = info_dict['target_name']
        answer = info_dict['answer'].split("\n")[0].split(".")[0].lower()

        if groundtruth == answer:
            correct += 1
        
        # line_dict['effect_type'] = info_dict['effect_type']
        # line_dict['groundtruth'] = groundtruth
        # line_dict['true_rate'] = true_rate
        # line_dict['false_rate'] = false_rate
        # line_dict['answers'] = info_dict['answer']

        # f_write.write(line_dict)
        # total_correct.append(true_rate)
        # break


print(correct / total_correct)

