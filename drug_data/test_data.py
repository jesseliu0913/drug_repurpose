import os
import json
import pandas as pd


Disease_Path = "../PrimeKG/disease_features.csv"
Drug_Path = "../PrimeKG/drug_features.csv"
df_disease = pd.read_csv(Disease_Path)
df_drug = pd.read_csv(Drug_Path)
# print(df_disease.columns)
# print(df_disease.head())
# print(df_drug.columns)
# print(df_drug.head())

Edge_Path = "../PrimeKG/edges.csv"
df_edge = pd.read_csv(Edge_Path)
ralation_type = set(list(df_edge['relation']))
"""
{'indication', 'bioprocess_bioprocess', 'phenotype_protein', 'pathway_protein', 'anatomy_protein_absent', 
'cellcomp_cellcomp', 'drug_protein', 'protein_protein', 'disease_phenotype_negative', 'molfunc_protein', 
'exposure_protein', 'exposure_exposure', 'disease_phenotype_positive', 'anatomy_protein_present', 'disease_disease', 
'drug_effect', 'exposure_cellcomp', 'contraindication', 'cellcomp_protein', 'disease_protein', 'exposure_disease', 
'exposure_molfunc', 'exposure_bioprocess', 'phenotype_phenotype', 'pathway_pathway', 'anatomy_anatomy', 'drug_drug', 
'molfunc_molfunc', 'bioprocess_protein', 'off-label use'}
"""

