import os
import ast
import jsonlines
import pandas as pd


negative_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/train_data_negative.csv")
nodes_data = pd.read_csv("/playpen/jesse/drug_repurpose/PrimeKG/nodes.csv")
file_path_phenotype = "./data/dd_phenotype_negative.jsonl"
file_path_protein = "./data/dd_protein_negative.jsonl"
f_write_pheno =  jsonlines.open(file_path_phenotype, "a")
f_write_gene =  jsonlines.open(file_path_protein, "a")

for index, row in negative_data.iterrows():
    disease_name = row.disease_name
    drug_name = row.drug_name

    phenotype = ast.literal_eval(row['related_phenotypes'])[0]
    protein = ast.literal_eval(row['related_proteins'])[0]
    phenotypes = nodes_data[nodes_data['node_index'] == phenotype][['node_name', 'node_type']]
    proteins = nodes_data[nodes_data['node_index'] == protein][['node_name', 'node_type']]

    if not phenotypes.empty:
        phenotype_name = phenotypes.iloc[0]['node_name']
        phenotype_type = phenotypes.iloc[0]['node_type']

        input_pheno = f"Given that {disease_name} is associated with {phenotype_name}, which is not treatable by {drug_name}, it follows that {disease_name} is a contraindication for {drug_name}."
        line_dict_pn = {"drug_name": drug_name, "disease_name": disease_name, "phenotype_name": phenotype_name, "prompt": input_pheno}
        f_write_pheno.write(line_dict_pn)
    
    if not proteins.empty:
        protein_name = proteins.iloc[0]['node_name']
        protein_type = proteins.iloc[0]['node_type']

        input_gene = f"Given the association between {disease_name} and {protein_name}, and the fact that {drug_name} not target {protein_name}, so {disease_name} is a contraindication for {drug_name}."
        line_dict_gene = {"drug_name": drug_name, "disease_name": disease_name, "gene_name": protein_name, "prompt": input_gene}
        f_write_gene.write(line_dict_gene)
    

    

