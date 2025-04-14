import os
import json
import jsonlines


train_data = open("/playpen/jesse/drug_repurpose/split_data/data/train.jsonl", "r", encoding="utf-8")
file_path = "/playpen/jesse/drug_repurpose/split_data/data/train_modify.jsonl"
train_data = [json.loads(line) for line in train_data]

with jsonlines.open(file_path, "a") as f_write:
    for td in train_data:
        prompt = td["prompt"]
        drug_name = td["drug_name"]
        disease_name = td["disease_name"]
        answer = td["answer"]

        question = f"Is {disease_name} an indication for {drug_name}?"
        if "no" in answer.lower():
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: NO"
        else:
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: YES"

        td["prompt"] = prefix
        td["reasoning"] = prompt

        f_write.write(td)
        