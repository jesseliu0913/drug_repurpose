import json
import pandas as pd

# with open("../drug_data/data/disease_feature.json", "r", encoding="utf-8") as file:
#     disease_data = json.load(file)
# subset_data = dict(list(disease_data.items())[:400])

# with open("disease_feature_subset.json", "w", encoding="utf-8") as outfile:
#     json.dump(subset_data, outfile, ensure_ascii=False, indent=4)
# for dk in disease_keys:
#     disease_content = disease_data[dk]
#     contraindication = disease_content['contraindication'][-1]
#     indication = disease_content['indication'][-1]
#     print("disease name", dk)
#     print(contraindication)
#     print(indication)
#     break

file_path = "/playpen/jesse/drug_repurpose/PharmacotherapyDB/catalog.xlsx"
df = pd.read_excel(file_path)
print(df.head())