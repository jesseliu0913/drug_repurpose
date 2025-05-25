import pandas as pd
import numpy as np
from collections import defaultdict

NODE_FILE = "/playpen/jesse/drug_repurpose/PrimeKG/nodes.csv"
EDGE_FILE = "/playpen/jesse/drug_repurpose/PrimeKG/edges.csv"

df_node = pd.read_csv(NODE_FILE)
df_edges = pd.read_csv(EDGE_FILE)

node_type_map = dict(zip(df_node['node_index'], df_node['node_type']))
node_name_map = dict(zip(df_node['node_index'], df_node['node_name']))
max_paths = 2000
disease_nodes = set(df_node[df_node['node_type'] == 'disease']['node_index'])
drug_nodes = set(df_node[df_node['node_type'] == 'drug']['node_index'])
gene_protein_nodes = set(df_node[df_node['node_type'] == 'gene/protein']['node_index'])
phenotype_nodes = set(df_node[df_node['node_type'] == 'effect/phenotype']['node_index'])
exposure_nodes = set(df_node[df_node['node_type'] == 'exposure']['node_index'])
bioprocess_nodes = set(df_node[df_node['node_type'] == 'biological_process']['node_index'])
molfunc_nodes = set(df_node[df_node['node_type'] == 'molecular_function']['node_index'])
cellcomp_nodes = set(df_node[df_node['node_type'] == 'cellular_component']['node_index'])

def find_established_disease_drug_pairs():
    """Identify disease-drug pairs that already have established relationships
    (indication, contraindication, off-label use)"""
    established_pairs = set()
    
    disease_drug_edges = df_edges[
        ((df_edges['x_index'].isin(disease_nodes)) & 
         (df_edges['y_index'].isin(drug_nodes)) & 
         (df_edges['relation'].isin(['contraindication'])))
    ]
    
    for _, row in disease_drug_edges.iterrows():
        established_pairs.add((row['x_index'], row['y_index'], row['relation']))
    
    drug_disease_edges = df_edges[
        ((df_edges['x_index'].isin(drug_nodes)) & 
         (df_edges['y_index'].isin(disease_nodes)) & 
         (df_edges['relation'].isin(['contraindication'])))
    ]
    
    for _, row in drug_disease_edges.iterrows():
        established_pairs.add((row['y_index'], row['x_index'], row['relation']))
    
    established_relations = {}
    for disease, drug, relation in established_pairs:
        established_relations[(disease, drug)] = relation
    
    return established_relations

def find_disease_protein_drug_paths(established_pairs):
    indirect_paths = []
    
    disease_protein_map = defaultdict(set)
    protein_drug_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(gene_protein_nodes)) & 
        (df_edges['relation'] == 'disease_protein')
    ].iterrows():
        disease_protein_map[row['x_index']].add(row['y_index'])
    

    for _, row in df_edges[
        (df_edges['x_index'].isin(gene_protein_nodes)) & 
        (df_edges['y_index'].isin(drug_nodes)) & 
        (df_edges['relation'] == 'drug_protein')
    ].iterrows():
        protein_drug_map[row['x_index']].add(row['y_index'])
    
    for (disease, drug), relation in established_pairs.items():
        for protein in disease_protein_map.get(disease, set()):
            if drug not in protein_drug_map.get(protein, set()):
                indirect_paths.append({
                    'disease_index': disease,
                    'disease_name': node_name_map[disease],
                    'protein_index': protein,
                    'protein_name': node_name_map[protein],
                    'drug_index': drug,
                    'drug_name': node_name_map[drug],
                    'original_relation': relation,
                    'path_type': 'disease-protein-drug'
                })
    
                if len(indirect_paths) >= max_paths:
                    return indirect_paths
    
    return indirect_paths

def find_disease_phenotype_protein_drug_paths(established_pairs):
    complex_paths = []
    
    disease_phenotype_map = defaultdict(set)
    phenotype_protein_map = defaultdict(set)
    protein_drug_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(phenotype_nodes)) & 
        (df_edges['relation'] == 'disease_phenotype_positive')
    ].iterrows():
        disease_phenotype_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(phenotype_nodes)) & 
        (df_edges['y_index'].isin(gene_protein_nodes)) & 
        (df_edges['relation'] == 'phenotype_protein')
    ].iterrows():
        phenotype_protein_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(gene_protein_nodes)) & 
        (df_edges['y_index'].isin(drug_nodes)) & 
        (df_edges['relation'] == 'drug_protein')
    ].iterrows():
        protein_drug_map[row['x_index']].add(row['y_index'])

    for (disease, drug), relation in established_pairs.items():
        for phenotype in disease_phenotype_map.get(disease, set()):
            for protein in phenotype_protein_map.get(phenotype, set()):
                if drug not in protein_drug_map.get(protein, set()):
                    complex_paths.append({
                        'disease_index': disease,
                        'disease_name': node_name_map[disease],
                        'phenotype_index': phenotype,
                        'phenotype_name': node_name_map[phenotype],
                        'protein_index': protein,
                        'protein_name': node_name_map[protein],
                        'drug_index': drug,
                        'drug_name': node_name_map[drug],
                        'original_relation': relation,
                        'path_type': 'disease-phenotype-protein-drug'
                    })

                    if len(complex_paths) >= max_paths:
                        return complex_paths
    
    return complex_paths

def find_disease_disease_drug_paths(established_pairs):
    disease_disease_paths = []
    
    disease_disease_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(disease_nodes)) & 
        (df_edges['relation'] == 'disease_disease')
    ].iterrows():
        disease_disease_map[row['x_index']].add(row['y_index'])
    
    for (disease1, drug), relation in established_pairs.items():
        for disease2 in disease_disease_map.get(disease1, set()):
            if (disease2, drug) not in established_pairs:
                disease_disease_paths.append({
                    'disease1_index': disease1,
                    'disease1_name': node_name_map[disease1],
                    'disease2_index': disease2,
                    'disease2_name': node_name_map[disease2],
                    'drug_index': drug,
                    'drug_name': node_name_map[drug],
                    'original_relation': relation,
                    'path_type': 'disease-disease-drug'
                })
                if len(disease_disease_paths) >= max_paths:
                    return disease_disease_paths
    
    return disease_disease_paths

def find_disease_exposure_gene_drug_paths(established_pairs):
    exposure_paths = []
    
    disease_exposure_map = defaultdict(set)
    exposure_gene_map = defaultdict(set)
    gene_drug_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(exposure_nodes)) & 
        (df_edges['relation'] == 'exposure_disease')
    ].iterrows():
        disease_exposure_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(exposure_nodes)) & 
        (df_edges['y_index'].isin(gene_protein_nodes)) & 
        (df_edges['relation'] == 'exposure_protein')
    ].iterrows():
        exposure_gene_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(gene_protein_nodes)) & 
        (df_edges['y_index'].isin(drug_nodes)) & 
        (df_edges['relation'] == 'drug_protein')
    ].iterrows():
        gene_drug_map[row['x_index']].add(row['y_index'])
    
    for (disease, drug), relation in established_pairs.items():
        for exposure in disease_exposure_map.get(disease, set()):
            for gene in exposure_gene_map.get(exposure, set()):
                if drug not in gene_drug_map.get(gene, set()):
                    exposure_paths.append({
                        'disease_index': disease,
                        'disease_name': node_name_map[disease],
                        'exposure_index': exposure,
                        'exposure_name': node_name_map[exposure],
                        'gene_index': gene,
                        'gene_name': node_name_map[gene],
                        'drug_index': drug,
                        'drug_name': node_name_map[drug],
                        'original_relation': relation,
                        'path_type': 'disease-exposure-gene-drug'
                    })

                    if len(exposure_paths) >= max_paths:
                        return exposure_paths
    
    return exposure_paths

def find_disease_exposure_bioprocess_gene_drug_paths(established_pairs):
    bioprocess_paths = []
    
    disease_exposure_map = defaultdict(set)
    exposure_bioprocess_map = defaultdict(set)
    bioprocess_gene_map = defaultdict(set)
    gene_drug_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(exposure_nodes)) & 
        (df_edges['relation'] == 'exposure_disease')
    ].iterrows():
        disease_exposure_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(exposure_nodes)) & 
        (df_edges['y_index'].isin(bioprocess_nodes)) & 
        (df_edges['relation'] == 'exposure_bioprocess')
    ].iterrows():
        exposure_bioprocess_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(bioprocess_nodes)) & 
        (df_edges['y_index'].isin(gene_protein_nodes)) & 
        (df_edges['relation'] == 'bioprocess_protein')
    ].iterrows():
        bioprocess_gene_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(gene_protein_nodes)) & 
        (df_edges['y_index'].isin(drug_nodes)) & 
        (df_edges['relation'] == 'drug_protein')
    ].iterrows():
        gene_drug_map[row['x_index']].add(row['y_index'])
    
    for (disease, drug), relation in established_pairs.items():
        for exposure in disease_exposure_map.get(disease, set()):
            for bioprocess in exposure_bioprocess_map.get(exposure, set()):
                for gene in bioprocess_gene_map.get(bioprocess, set()):
                    if drug not in gene_drug_map.get(gene, set()):
                        bioprocess_paths.append({
                            'disease_index': disease,
                            'disease_name': node_name_map[disease],
                            'exposure_index': exposure,
                            'exposure_name': node_name_map[exposure],
                            'bioprocess_index': bioprocess,
                            'bioprocess_name': node_name_map[bioprocess],
                            'gene_index': gene,
                            'gene_name': node_name_map[gene],
                            'drug_index': drug,
                            'drug_name': node_name_map[drug],
                            'original_relation': relation,
                            'path_type': 'disease-exposure-bioprocess-gene-drug'
                        })

                        if len(bioprocess_paths) >= max_paths:
                            return bioprocess_paths
    
    return bioprocess_paths

def find_disease_exposure_molfunc_gene_drug_paths(established_pairs):
    molfunc_paths = []
    
    disease_exposure_map = defaultdict(set)
    exposure_molfunc_map = defaultdict(set)
    molfunc_gene_map = defaultdict(set)
    gene_drug_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(exposure_nodes)) & 
        (df_edges['relation'] == 'exposure_disease')
    ].iterrows():
        disease_exposure_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(exposure_nodes)) & 
        (df_edges['y_index'].isin(molfunc_nodes)) & 
        (df_edges['relation'] == 'exposure_molfunc')
    ].iterrows():
        exposure_molfunc_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(molfunc_nodes)) & 
        (df_edges['y_index'].isin(gene_protein_nodes)) & 
        (df_edges['relation'] == 'molfunc_protein')
    ].iterrows():
        molfunc_gene_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(gene_protein_nodes)) & 
        (df_edges['y_index'].isin(drug_nodes)) & 
        (df_edges['relation'] == 'drug_protein')
    ].iterrows():
        gene_drug_map[row['x_index']].add(row['y_index'])
    
    for (disease, drug), relation in established_pairs.items():
        for exposure in disease_exposure_map.get(disease, set()):
            for molfunc in exposure_molfunc_map.get(exposure, set()):
                for gene in molfunc_gene_map.get(molfunc, set()):
                    if drug not in gene_drug_map.get(gene, set()):
                        molfunc_paths.append({
                            'disease_index': disease,
                            'disease_name': node_name_map[disease],
                            'exposure_index': exposure,
                            'exposure_name': node_name_map[exposure],
                            'molfunc_index': molfunc,
                            'molfunc_name': node_name_map[molfunc],
                            'gene_index': gene,
                            'gene_name': node_name_map[gene],
                            'drug_index': drug,
                            'drug_name': node_name_map[drug],
                            'original_relation': relation,
                            'path_type': 'disease-exposure-molfunc-gene-drug'
                        })
                        
                        if len(molfunc_paths) >= max_paths:
                            return molfunc_paths
    return molfunc_paths

def find_disease_exposure_cellcomp_gene_drug_paths(established_pairs):
    cellcomp_paths = []

    disease_exposure_map = defaultdict(set)
    exposure_cellcomp_map = defaultdict(set)
    cellcomp_gene_map = defaultdict(set)
    gene_drug_map = defaultdict(set)
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(disease_nodes)) & 
        (df_edges['y_index'].isin(exposure_nodes)) & 
        (df_edges['relation'] == 'exposure_disease')
    ].iterrows():
        disease_exposure_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(exposure_nodes)) & 
        (df_edges['y_index'].isin(cellcomp_nodes)) & 
        (df_edges['relation'] == 'exposure_cellcomp')
    ].iterrows():
        exposure_cellcomp_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(cellcomp_nodes)) & 
        (df_edges['y_index'].isin(gene_protein_nodes)) & 
        (df_edges['relation'] == 'cellcomp_protein')
    ].iterrows():
        cellcomp_gene_map[row['x_index']].add(row['y_index'])
    
    for _, row in df_edges[
        (df_edges['x_index'].isin(gene_protein_nodes)) & 
        (df_edges['y_index'].isin(drug_nodes)) & 
        (df_edges['relation'] == 'drug_protein')
    ].iterrows():
        gene_drug_map[row['x_index']].add(row['y_index'])
    
    for (disease, drug), relation in established_pairs.items():
        for exposure in disease_exposure_map.get(disease, set()):
            for cellcomp in exposure_cellcomp_map.get(exposure, set()):
                for gene in cellcomp_gene_map.get(cellcomp, set()):
                    if drug not in gene_drug_map.get(gene, set()):
                        cellcomp_paths.append({
                            'disease_index': disease,
                            'disease_name': node_name_map[disease],
                            'exposure_index': exposure,
                            'exposure_name': node_name_map[exposure],
                            'cellcomp_index': cellcomp,
                            'cellcomp_name': node_name_map[cellcomp],
                            'gene_index': gene,
                            'gene_name': node_name_map[gene],
                            'drug_index': drug,
                            'drug_name': node_name_map[drug],
                            'original_relation': relation,
                            'path_type': 'disease-exposure-cellcomp-gene-drug'
                        })

                        if len(cellcomp_paths) >= max_paths:
                            return cellcomp_paths
    
    return cellcomp_paths


print("Finding established disease-drug pairs...")
established_pairs = find_established_disease_drug_pairs()
print(f"Found {len(established_pairs)} established disease-drug pairs")

print("Finding paths between established disease-drug pairs...")
disease_protein_drug_paths = find_disease_protein_drug_paths(established_pairs)
disease_phenotype_paths = find_disease_phenotype_protein_drug_paths(established_pairs)
disease_disease_paths = find_disease_disease_drug_paths(established_pairs)
disease_exposure_gene_paths = find_disease_exposure_gene_drug_paths(established_pairs)
disease_exposure_bioprocess_paths = find_disease_exposure_bioprocess_gene_drug_paths(established_pairs)
disease_exposure_molfunc_paths = find_disease_exposure_molfunc_gene_drug_paths(established_pairs)
disease_exposure_cellcomp_paths = find_disease_exposure_cellcomp_gene_drug_paths(established_pairs)


import os
if not os.path.exists('./negative_path'):
    os.makedirs('./negative_path')

pd.DataFrame(list(established_pairs.items()), columns=['disease_drug_pair', 'relation']).to_csv('./path_data/established_disease_drug_pairs.csv', index=False)
pd.DataFrame(disease_protein_drug_paths).to_csv('./negative_path/disease_protein_drug_paths.csv', index=False)
pd.DataFrame(disease_phenotype_paths).to_csv('./negative_path/disease_phenotype_protein_drug_paths.csv', index=False)
pd.DataFrame(disease_disease_paths).to_csv('./negative_path/disease_disease_drug_paths.csv', index=False)
pd.DataFrame(disease_exposure_gene_paths).to_csv('./negative_path/disease_exposure_gene_drug_paths.csv', index=False)
pd.DataFrame(disease_exposure_bioprocess_paths).to_csv('./negative_path/disease_exposure_bioprocess_gene_drug_paths.csv', index=False)
pd.DataFrame(disease_exposure_molfunc_paths).to_csv('./negative_path/disease_exposure_molfunc_gene_drug_paths.csv', index=False)
pd.DataFrame(disease_exposure_cellcomp_paths).to_csv('./negative_path/disease_exposure_cellcomp_gene_drug_paths.csv', index=False)

print(f"Disease-protein-drug paths for established pairs: {len(disease_protein_drug_paths)}")
print(f"Disease-phenotype-protein-drug paths for established pairs: {len(disease_phenotype_paths)}")
print(f"Disease-disease-drug paths for established pairs: {len(disease_disease_paths)}")
print(f"Disease-exposure-gene-drug paths for established pairs: {len(disease_exposure_gene_paths)}")
print(f"Disease-exposure-bioprocess-gene-drug paths for established pairs: {len(disease_exposure_bioprocess_paths)}")
print(f"Disease-exposure-molfunc-gene-drug paths for established pairs: {len(disease_exposure_molfunc_paths)}")
print(f"Disease-exposure-cellcomp-gene-drug paths for established pairs: {len(disease_exposure_cellcomp_paths)}")

total_paths = (len(disease_protein_drug_paths) + 
               len(disease_phenotype_paths) + 
               len(disease_disease_paths) + 
               len(disease_exposure_gene_paths) + 
               len(disease_exposure_bioprocess_paths) + 
               len(disease_exposure_molfunc_paths) + 
               len(disease_exposure_cellcomp_paths))

print(f"Total alternative paths for established disease-drug pairs: {total_paths}")


pairs_with_paths = set()

for path in disease_protein_drug_paths + disease_phenotype_paths + disease_disease_paths + \
           disease_exposure_gene_paths + disease_exposure_bioprocess_paths + \
           disease_exposure_molfunc_paths + disease_exposure_cellcomp_paths:
    if 'disease_index' in path and 'drug_index' in path:
        pairs_with_paths.add((path['disease_index'], path['drug_index']))
    elif 'disease1_index' in path:  
        pairs_with_paths.add((path['disease1_index'], path['drug_index']))

print(f"Number of established pairs with at least one alternative path: {len(pairs_with_paths)}")
print(f"Percentage of established pairs with alternative paths: {len(pairs_with_paths)/len(established_pairs)*100:.2f}%")