import os
import jsonlines
import pandas as pd


ddphenotype_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/ddphenotype.csv")
ddprotein_data = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/ddprotein.csv")

file_path_phenotype = "./data/ddphenotype.jsonl"
with jsonlines.open(file_path_phenotype, "a") as f_write:
    for index, row in ddphenotype_data.iterrows():
        disease_name = row.disease_name
        phenotype_name = row.phenotype_name
        drug_name = row.drug_name

        input_text = f"Given that {disease_name} is associated with {phenotype_name}, which is treatable by {drug_name}, it follows that {disease_name} is an indication for {drug_name}."
        line_dict = {"drug_name": drug_name, "disease_name": disease_name, "phenotype_name": phenotype_name, "prompt": input_text}
        f_write.write(line_dict)

file_path_protein = "./data/ddgene.jsonl"
with jsonlines.open(file_path_protein, "a") as f_write:
    for index, row in ddprotein_data.iterrows():
        disease_name = row.disease_name
        gene_name = row.gene_name
        drug_name = row.drug_name
        
        input_text = f"Given the association between {disease_name} and {gene_name}, and the fact that {drug_name} targets {gene_name}, {disease_name} is an indication for {drug_name}."
        line_dict = {"drug_name": drug_name, "disease_name": disease_name, "gene_name": gene_name, "prompt": input_text}
        f_write.write(line_dict)

