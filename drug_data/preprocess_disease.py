import os
import json
import pandas as pd


Disease_Path = "../PrimeKG/disease_features.csv"
Drug_Path = "../PrimeKG/drug_features.csv"
Edge_Path = "../PrimeKG/edges.csv"
Node_Path = "../PrimeKG/nodes.csv"

ralation_disease = {"disease_disease", "disease_phenotype_positive", "disease_phenotype_negative", "disease_protein", "indication", "contraindication", "off-label use", "exposure_disease"}
df_edge = pd.read_csv(Edge_Path)
df_disease = pd.read_csv(Disease_Path)
df_drug = pd.read_csv(Drug_Path)
df_node = pd.read_csv(Node_Path)

disease_edges = df_edge[df_edge['relation'].isin(ralation_disease)].copy()
name_lookup = dict(zip(df_node['node_index'], df_node['node_name']))
type_lookup = dict(zip(df_node['node_index'], df_node['node_type']))

disease_edges['x_name'] = disease_edges['x_index'].map(name_lookup)
disease_edges['y_name'] = disease_edges['y_index'].map(name_lookup)
disease_edges['x_type'] = disease_edges['x_index'].map(type_lookup)
disease_edges['y_type'] = disease_edges['y_index'].map(type_lookup)

disease_dict = {}
flag = None
for index, row in disease_edges.iterrows():
    if row['x_type'] == "disease":
        flag = "x_index"
        disease_name = row['x_name']
    elif row['y_type'] == "disease":
        flag = "y_index"
        disease_name = row['y_name']
    else:
        print(f"Disease not exits!!! with {index}")
        break
    
    if disease_name not in list(disease_dict.keys()):
        disease_dict[disease_name] = {}
    
    if flag == "x_index":
        if row["relation"] not in list(disease_dict[disease_name].keys()):
            disease_dict[disease_name][row["relation"]] = [row['y_name']]
        else:
            disease_dict[disease_name][row["relation"]].append(row['y_name'])
    elif flag == "y_index":
        if row["relation"] not in list(disease_dict[disease_name].keys()):
            disease_dict[disease_name][row["relation"]] = [row['x_name']]
        else:
            disease_dict[disease_name][row["relation"]].append(row['x_name'])

print(len(disease_dict.keys()))
with open("./data/disease_feature.json", "w") as file:
    json.dump(disease_dict, file)

