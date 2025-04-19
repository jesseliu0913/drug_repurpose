import os
import json


def extract_drug_disease_pairs(file_path):
    drug_disease_pairs = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            drug_disease_pairs.add((data['drug_name'], data['disease_name']))
    return drug_disease_pairs

file_A = "/playpen/jesse/drug_repurpose/eval_results/uncertainty_results/llama32_1b/cot.jsonl"
file_B = "/playpen/jesse/drug_repurpose/eval_results/results/llama32_1b/cot.jsonl"

pairs_A = extract_drug_disease_pairs(file_A)
pairs_B = extract_drug_disease_pairs(file_B)

exclusive_to_A = pairs_A - pairs_B

print("Tuples in A but not in B:")
for pair in exclusive_to_A:
    print(pair)
