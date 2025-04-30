import pandas as pd
from collections import defaultdict

path_files = {
    'protein': './path_data/disease_protein_drug_paths.csv',
    'phenotype': './path_data/disease_phenotype_protein_drug_paths.csv',
    'disease': './path_data/disease_disease_drug_paths.csv',
    'exposure_gene': './path_data/disease_exposure_gene_drug_paths.csv',
    'exposure_bioprocess': './path_data/disease_exposure_bioprocess_gene_drug_paths.csv',
    'exposure_molfunc': './path_data/disease_exposure_molfunc_gene_drug_paths.csv',
    'exposure_cellcomp': './path_data/disease_exposure_cellcomp_gene_drug_paths.csv'
}

established_pairs = pd.read_csv('./path_data/established_disease_drug_pairs.csv')
established_dict = dict(zip(zip(established_pairs['disease_index'], established_pairs['drug_index']), 
                           established_pairs['relation']))

path_type_columns = list(path_files.keys())
pairs_path_types = defaultdict(lambda: defaultdict(int))
relation_path_types = defaultdict(lambda: defaultdict(int))
pairs_with_paths = set()

for path_type, file_path in path_files.items():
    paths_df = pd.read_csv(file_path)
    
    if path_type == 'disease':
        for _, row in paths_df.iterrows():
            pair = (row['disease1_index'], row['drug_index'])
            pairs_path_types[pair][path_type] += 1
            pairs_with_paths.add(pair)
            
            relation = row['original_relation']
            relation_path_types[relation][path_type] += 1
    else:
        for _, row in paths_df.iterrows():
            pair = (row['disease_index'], row['drug_index'])
            pairs_path_types[pair][path_type] += 1
            pairs_with_paths.add(pair)
            
            relation = row['original_relation']
            relation_path_types[relation][path_type] += 1

pair_data = []
for pair, path_counts in pairs_path_types.items():
    disease_index, drug_index = pair
    
    total_paths = sum(path_counts.values())
    
    path_percentages = {}
    for path_type in path_type_columns:
        path_count = path_counts.get(path_type, 0)
        percentage = (path_count / total_paths * 100) if total_paths > 0 else 0
        path_percentages[f"{path_type}_percent"] = percentage
    
    path_type_diversity = sum(1 for pt in path_type_columns if path_counts.get(pt, 0) > 0)
    
    pair_data.append({
        'disease_index': disease_index,
        'disease_name': established_pairs[
            (established_pairs['disease_index'] == disease_index) & 
            (established_pairs['drug_index'] == drug_index)
        ]['disease_name'].values[0],
        'drug_index': drug_index, 
        'drug_name': established_pairs[
            (established_pairs['disease_index'] == disease_index) & 
            (established_pairs['drug_index'] == drug_index)
        ]['drug_name'].values[0],
        'relation': established_dict.get((disease_index, drug_index), "unknown"),
        'total_paths': total_paths,
        'path_type_diversity': path_type_diversity,
        **path_counts,
        **path_percentages
    })

pairs_analysis_df = pd.DataFrame(pair_data)

for path_type in path_type_columns:
    if path_type not in pairs_analysis_df.columns:
        pairs_analysis_df[path_type] = 0
    if f"{path_type}_percent" not in pairs_analysis_df.columns:
        pairs_analysis_df[f"{path_type}_percent"] = 0

high_diversity_pairs = pairs_analysis_df.sort_values(['path_type_diversity', 'total_paths'], ascending=False).head(20)
print("\nTop 20 disease-drug pairs with highest path type diversity:")
for _, row in high_diversity_pairs.iterrows():
    print(f"Disease: {row['disease_name']}, Drug: {row['drug_name']}")
    print(f"  Relation: {row['relation']}")
    print(f"  Path diversity: {row['path_type_diversity']} types, Total paths: {row['total_paths']}")
    for path_type in path_type_columns:
        if row[path_type] > 0:
            print(f"    {path_type}: {row[path_type]} paths ({row[f'{path_type}_percent']:.1f}%)")
    print()

pairs_analysis_df.to_csv('./path_data/disease_drug_pair_path_analysis.csv', index=False)

path_counts = {}
for path_type in path_type_columns:
    paths_df = pd.read_csv(path_files[path_type])
    path_counts[path_type] = len(paths_df)

total_paths = sum(path_counts.values())
print("\nOverall percentage distribution of path types:")
for path_type, count in sorted(path_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_paths) * 100 if total_paths > 0 else 0
    print(f"{path_type}: {count} paths ({percentage:.2f}% of all paths)")

avg_percentages = {}
for path_type in path_type_columns:
    percent_col = f"{path_type}_percent"
    avg_percentages[path_type] = pairs_analysis_df[percent_col].mean()

print("\nAverage percentage of each path type per disease-drug pair:")
for path_type, avg_percent in sorted(avg_percentages.items(), key=lambda x: x[1], reverse=True):
    print(f"{path_type}: {avg_percent:.2f}%")

print("\nPath type distribution by relation type:")
path_type_totals = defaultdict(int)

for relation, path_counts in relation_path_types.items():
    total_for_relation = sum(path_counts.values())
    path_type_totals[relation] = total_for_relation
    
    print(f"\nRelation: {relation} (Total paths: {total_for_relation})")
    for path_type, count in sorted(path_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_for_relation) * 100 if total_for_relation > 0 else 0
        print(f"  {path_type}: {count} paths ({percentage:.2f}%)")

relation_analysis_data = []
for relation, path_counts in relation_path_types.items():
    total_paths = sum(path_counts.values())
    
    path_percentages = {}
    for path_type in path_type_columns:
        path_count = path_counts.get(path_type, 0)
        percentage = (path_count / total_paths * 100) if total_paths > 0 else 0
        path_percentages[f"{path_type}_percent"] = percentage
    
    relation_analysis_data.append({
        'relation': relation,
        'total_paths': total_paths,
        **path_counts,
        **path_percentages
    })

relation_analysis_df = pd.DataFrame(relation_analysis_data)
relation_analysis_df.to_csv('./path_data/relation_path_analysis.csv', index=False)

relation_groups = pairs_analysis_df.groupby('relation')
print("\nPath diversity by relation type:")
for relation, group in relation_groups:
    avg_diversity = group['path_type_diversity'].mean()
    max_diversity = group['path_type_diversity'].max()
    count = len(group)
    print(f"{relation}: {count} pairs, avg diversity {avg_diversity:.2f}, max diversity {max_diversity}")

most_common_paths = pairs_analysis_df.groupby('disease_name').size().sort_values(ascending=False).head(20)
print("\nTop 20 diseases with most alternative paths to drugs:")
for disease, count in most_common_paths.items():
    print(f"{disease}: {count} drug connections")
    
most_common_drug_paths = pairs_analysis_df.groupby('drug_name').size().sort_values(ascending=False).head(20)
print("\nTop 20 drugs with most alternative paths to diseases:")
for drug, count in most_common_drug_paths.items():
    print(f"{drug}: {count} disease connections")