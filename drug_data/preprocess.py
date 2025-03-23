import os
import json
import pandas as pd


Disease_Path = "../PrimeKG/disease_features.csv"
Drug_Path = "../PrimeKG/drug_features.csv"
Edge_Path = "../PrimeKG/edges.csv"

df_edge = pd.read_csv(Edge_Path)
df_disease = pd.read_csv(Disease_Path)
df_drug = pd.read_csv(Drug_Path)


# if ralation_type == "indication":
indication_df = df_edge[df_edge["relation"] == "indication"]
print(indication_df.head())

