import os
import json
import pandas as pd


Disease_Path = "../PrimeKG/disease_features.csv"
Drug_Path = "../PrimeKG/drug_features.csv"
Edge_Path = "../PrimeKG/edges.csv"
Node_Path = "../PrimeKG/nodes.csv"

ralation_drugs = {"protein_protein", "phenotype_protein"}
df_edge = pd.read_csv(Edge_Path)
df_disease = pd.read_csv(Disease_Path)
df_drug = pd.read_csv(Drug_Path)
df_node = pd.read_csv(Node_Path)

drugs_edges = df_edge[df_edge['relation'].isin(ralation_drugs)].copy()
name_lookup = dict(zip(df_node['node_index'], df_node['node_name']))
type_lookup = dict(zip(df_node['node_index'], df_node['node_type']))

drugs_edges['x_name'] = drugs_edges['x_index'].map(name_lookup)
drugs_edges['y_name'] = drugs_edges['y_index'].map(name_lookup)
drugs_edges['x_type'] = drugs_edges['x_index'].map(type_lookup)
drugs_edges['y_type'] = drugs_edges['y_index'].map(type_lookup)


drug_dict = {}
flag = None
for index, row in drugs_edges.iterrows():
    if row['x_type'] == "gene/protein":
        flag = "x_index"
        drug_name = row['x_name']
    elif row['y_type'] == "gene/protein":
        flag = "y_index"
        drug_name = row['y_name']
    else:
        print(f"Drug not exits!!! with {index}")
        break
    
    if drug_name not in list(drug_dict.keys()):
        drug_dict[drug_name] = {}
    
    if flag == "x_index":
        if row["relation"] not in list(drug_dict[drug_name].keys()):
            drug_dict[drug_name][row["relation"]] = [row['y_name']]
        else:
            drug_dict[drug_name][row["relation"]].append(row['y_name'])
    elif flag == "y_index":
        if row["relation"] not in list(drug_dict[drug_name].keys()):
            drug_dict[drug_name][row["relation"]] = [row['x_name']]
        else:
            drug_dict[drug_name][row["relation"]].append(row['x_name'])

print(len(drug_dict.keys()))
with open("./data/gene_feature.json", "w") as file:
    json.dump(drug_dict, file)

