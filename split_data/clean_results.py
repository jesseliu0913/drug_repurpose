import os

# drug,pgrep -u $(whoami) -a python,disease,hypercalcemia disease

import os
import json

folder_path = "/playpen/jesse/drug_repurpose/split_data/results/llama32_1b"

for filename in os.listdir(folder_path):
    if filename.endswith(".jsonl"):
        full_path = os.path.join(folder_path, filename)
        
        with open(full_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        filtered_lines = []
        for line in lines:
            try:
                data = json.loads(line)
                if not (data.get("drug_name") == "Alprazolam" and data.get("disease_name") == "anxiety disorder"):
                    filtered_lines.append(line)
            except json.JSONDecodeError:
                continue

        with open(full_path, 'w', encoding='utf-8') as file:
            file.writelines(filtered_lines)

print("Done removing matching entries.")
