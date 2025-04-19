import os

# drug,disease,hypercalcemia disease

import os
import json

<<<<<<< HEAD:split_data/clean_results.py
folder_path = "/playpen/jesse/drug_repurpose/split_data/results/qwq_32b"
=======
folder_path = "/playpen/jesse/drug_repurpose/split_data/results/gemini"
>>>>>>> 31929f16965b2c7202453abdb21937f8f7cd5ace:eval_results/clean_results.py

for filename in os.listdir(folder_path):
    if filename.endswith(".jsonl"):
        full_path = os.path.join(folder_path, filename)
        
        with open(full_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        filtered_lines = []
        for line in lines:
            try:
                data = json.loads(line)
                if not (data.get("drug_name") == "Calcitriol" and data.get("disease_name") == "hypercalcemia disease"):
                    filtered_lines.append(line)
            except json.JSONDecodeError:
                continue

        with open(full_path, 'w', encoding='utf-8') as file:
            file.writelines(filtered_lines)

print("Done removing matching entries.")
