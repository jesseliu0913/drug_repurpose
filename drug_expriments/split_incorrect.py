import os
import json
import csv


file_path = "./results/llama32_1b/output_DD1k.jsonl"
correct_csv_path = "./results/llama32_1b_split/output_DD1K_correct.csv"
incorrect_csv_path = "./results/llama32_1b_split/output_DD1K_incorrect.csv"

with open(correct_csv_path, mode="w", newline="") as correct_file, \
     open(incorrect_csv_path, mode="w", newline="") as incorrect_file:

    correct_writer = csv.writer(correct_file)
    incorrect_writer = csv.writer(incorrect_file)

    headers = ["drug_name", "target_name", "effect_type", "answer"]
    correct_writer.writerow(headers)
    incorrect_writer.writerow(headers)

    with open(file_path, 'r') as f_read:
        for line in f_read:
            info_dict = json.loads(line)

            if info_dict['effect_type'] == "contraindication":
                groundtruth = "no"
            elif info_dict['effect_type'] == "indication":
                groundtruth = "yes"
            else:
                groundtruth = info_dict['effect_type']

            drug_name = info_dict['drug_name']
            target_name = info_dict['target_name']
            effect_type = info_dict['effect_type']
            answer = info_dict['answer'].split("\n")[0].split(".")[0].lower()

            row = [drug_name, target_name, effect_type, answer]

            if groundtruth in ["yes", "no"] and groundtruth == answer:
                correct_writer.writerow(row)
            elif groundtruth in ["yes", "no"] and groundtruth != answer:
                incorrect_writer.writerow(row)

print(f"Data successfully written to {correct_csv_path} and {incorrect_csv_path}.")
