import pandas as pd
import random
import os

TEST_FILE = "./data_analysis/test_data_new.csv"
df_test = pd.read_csv(TEST_FILE)

test_pairs = set()
for _, row in df_test.iterrows():
    test_pairs.add((row['drug_index'], row['disease_index']))

direct_paths = pd.read_csv("path_data/direct_disease_drug_relations.csv")
disease_disease_paths = pd.read_csv("path_data/disease_disease_drug_paths.csv")
disease_phenotype_paths = pd.read_csv("path_data/disease_phenotype_protein_drug_paths.csv")
disease_protein_paths = pd.read_csv("path_data/disease_protein_drug_paths.csv")

def filter_test_pairs(df, drug_col, disease_col):
    return df[~df.apply(lambda row: (row[drug_col], row[disease_col]) in test_pairs, axis=1)].copy()

filtered_direct = filter_test_pairs(direct_paths, 'drug_index', 'disease_index')

filtered_disease_disease = filter_test_pairs(disease_disease_paths, 'drug_index', 'disease1_index')
filtered_disease_disease = filtered_disease_disease.rename(columns={'disease1_index': 'disease_index', 'disease1_name': 'disease_name'})

filtered_disease_phenotype = filter_test_pairs(disease_phenotype_paths, 'drug_index', 'disease_index')

filtered_disease_protein = filter_test_pairs(disease_protein_paths, 'drug_index', 'disease_index')

def standardize_columns(df, path_type):
    result_df = df.copy()
    
    essential_cols = []
    for col in ['disease_index', 'disease_name', 'drug_index', 'drug_name']:
        if col in result_df.columns:
            essential_cols.append(col)
        else:
            print(f"Warning: Column {col} not found in {path_type} dataframe")
    
    if 'relation' not in result_df.columns:
        result_df['relation'] = 'inferred_indication'
    else:
        essential_cols.append('relation')
    
    result_df['path_type'] = path_type
    
    if path_type == 'direct':
        result_df['path'] = "disease-" + result_df['relation'] + "-drug"
    elif path_type == 'disease-disease-drug':
        result_df['path'] = "disease-disease_disease-disease-indication-drug"
    elif path_type == 'disease-phenotype-protein-drug':
        result_df['path'] = "disease-disease_phenotype_positive-phenotype-phenotype_protein-protein-drug_protein-drug"
    elif path_type == 'disease-protein-drug':
        result_df['path'] = "disease-disease_protein-protein-drug_protein-drug"
    
    final_cols = essential_cols + ['path_type', 'path']
    return result_df[final_cols]

std_direct = standardize_columns(filtered_direct, 'direct')
std_disease_disease = standardize_columns(filtered_disease_disease, 'disease-disease-drug')
std_disease_phenotype = standardize_columns(filtered_disease_phenotype, 'disease-phenotype-protein-drug')
std_disease_protein = standardize_columns(filtered_disease_protein, 'disease-protein-drug')

all_paths = pd.concat([std_direct, std_disease_disease, std_disease_phenotype, std_disease_protein], ignore_index=True)

all_paths['pair'] = all_paths.apply(lambda row: (row['drug_index'], row['disease_index']), axis=1)
all_paths = all_paths.drop_duplicates(subset=['pair'])
all_paths = all_paths.drop(columns=['pair'])

indication_paths = all_paths[all_paths['relation'].isin(['indication', 'inferred_indication'])]
contraindication_paths = all_paths[all_paths['relation'] == 'contraindication']

random.seed(42)
if len(indication_paths) > 1000:
    indication_train = indication_paths.sample(n=1000, random_state=42)
else:
    indication_train = indication_paths

if len(contraindication_paths) > 1000:
    contraindication_train = contraindication_paths.sample(n=1000, random_state=42)
else:
    contraindication_train = contraindication_paths

training_set = pd.concat([indication_train, contraindication_train], ignore_index=True)

training_set.to_csv('drug_disease_training_set_with_paths.csv', index=False)

print(f"Total paths after removing test pairs: {len(all_paths)}")
print(f"Total indication paths: {len(indication_paths)}")
print(f"Total contraindication paths: {len(contraindication_paths)}")
print(f"Training set size: {len(training_set)}")
print(f"Indication pairs in training: {len(indication_train)}")
print(f"Contraindication pairs in training: {len(contraindication_train)}")