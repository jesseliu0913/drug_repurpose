import os
import json
import pandas as pd


# df_drug = pd.read_csv('./drug_effect_ungrouped.csv')
# df_drug = df_drug.drop_duplicates(subset=['drug_id'])

# file_path = "./unique_drug_disease.csv"
# df_drug.to_csv(file_path, index=False)

df_drug = pd.read_csv('./unique_drug_disease.csv')
print(len(df_drug))