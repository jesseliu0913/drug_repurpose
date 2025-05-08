import os
import json
import random


INPUT_PATH = "data/train.jsonl"
OUTPUT_PATH = "data/train_baseline.jsonl"

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    f.close()

def modify_data(data):
    modified_data = []
    for item in data:
        drug_name = item.get("drug_name", "")
        disease_name = item.get("disease_name", "")
        relation = item.get("relation", "")

        if relation == "indication":
            answer = "yes"
        elif relation == "contraindication":
            answer = "no"
    
        if drug_name and disease_name:
            prompt = prompt = f"Question: Is {drug_name} an indication for {disease_name}?\nAnswer: {answer}\n"
            item["prompt"] = prompt
            modified_data.append(item)
    return modified_data

if __name__ == "__main__":
    data = read_file(INPUT_PATH)
    print(f"Read {len(data)} lines from {INPUT_PATH}")

    modify_data(data)

    write_file(OUTPUT_PATH, data)
    print(f"Wrote {len(data)} lines to {OUTPUT_PATH}")