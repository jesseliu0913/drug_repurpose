import os
import json
import jsonlines
import pandas as pd


count = 0 
type = "gene"
pos_data = f"/playpen/jesse/drug_repurpose/split_data/gemini_select/output/negative/phenotype.jsonl"
raw_data = f"/playpen/jesse/drug_repurpose/split_data/data/dd_{type}_negative.jsonl"

f_write = jsonlines.open(f"/playpen/jesse/drug_repurpose/split_data/gemini_select/data/{type}_neg.jsonl", 'a')

with open(pos_data, 'r', encoding='utf-8') as f1, open(raw_data, 'r', encoding='utf-8') as f2:
    for output, raw in zip(f1, f2):
        output = json.loads(output)
        raw = json.loads(raw)
        answer = output['answer']
        drug_name = output['drug_name']
        disease_name = output['disease_name']
        assert drug_name == raw['drug_name'] and disease_name == raw['disease_name']

        if "NO" in answer:
            if type == "gene":
                line_dict = {"drug_name": drug_name, "disease_name": disease_name, "gene_name": raw['gene_name'], "answer": answer, "prompt": raw['prompt']}
                f_write.write(line_dict)
            else:
                line_dict = {"drug_name": drug_name, "disease_name": disease_name, "phenotype_name": raw['phenotype_name'], "answer": answer, "prompt": raw['prompt']}
                f_write.write(line_dict)

