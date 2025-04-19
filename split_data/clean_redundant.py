import json
import jsonlines
import pandas as pd


df = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
existing_pairs = set()

for index, row in df.iterrows():
    drug_disease_pair = (row[3], row[5])
    existing_pairs.add(drug_disease_pair)
    # print(f"Drug: {row[0]}, Disease: {row[1]}, Index: {index}")

problem_file = "/playpen/jesse/drug_repurpose/split_data/results/gemini/cot.jsonl"
# new_file = "/playpen/jesse/drug_repurpose/split_data/results/gemini/fraw_new.jsonl"
added_set = set()
# f_write = jsonlines.open(new_file, "a")
with open(problem_file, "r") as file:
    for line in file:
        line = json.loads(line)
        drug_name = line["drug_name"]
        disease_name = line["disease_name"]
        drug_disease_pair = (drug_name, disease_name)
        added_set.add(drug_disease_pair)
        # if drug_disease_pair in existing_pairs and drug_disease_pair not in added_set:
        #     added_set.add(drug_disease_pair)
            # f_write.write(line)

print(len(existing_pairs))
print(len(added_set))
print("set1 - set2:", added_set - existing_pairs)



