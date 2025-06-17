import pandas as pd

FILE_FOLDER = "/playpen/jesse/drug_repurpose/grpo_part_path/page_rank/"
file_path = f"{FILE_FOLDER}train_grpo_naive.csv"
df = pd.read_csv(file_path)

df['drug_index'] = df['drug_index'].astype(int)
df['disease_index'] = df['disease_index'].astype(int)

df = df.drop_duplicates(subset=['drug_index', 'disease_index'])
df.to_csv(f"{FILE_FOLDER}cleaned_train_grpo_naive.csv", index=False)

