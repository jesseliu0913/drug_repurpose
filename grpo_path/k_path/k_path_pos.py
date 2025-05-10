import os
import glob
import pandas as pd
import networkx as nx
from collections import defaultdict

test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test_pairs = set(zip(test.drug_index, test.disease_index))
print(f"Size of test_pairs: {len(test_pairs)}")

path_files = [f for f in glob.glob("../path_data/*.csv") if f.endswith("_paths.csv")]
print(f"Number of path files: {len(path_files)}")

all_paths = []

for fn in path_files:
    df = pd.read_csv(fn)
    print(f"Processing file: {fn}, Rows: {len(df)}")
    
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    if not dis_cols:
        print(f"  No disease index columns found in {fn}, skipping")
        continue
    
    initial_rows = len(df)
    df = df[~df.apply(lambda r: (r.drug_index, r[dis_cols[0]]) in test_pairs, axis=1)]
    filtered_rows = len(df)
    print(f"  Filtered from {initial_rows} to {filtered_rows} rows")
    
    if filtered_rows == 0:
        print(f"  No rows remain after filtering in {fn}")
        continue
    
    all_paths.append(df)

if not all_paths:
    print("No paths available after processing all files")
else:
    combined_df = pd.concat(all_paths, ignore_index=True)
    print(f"Total rows after combining all files: {len(combined_df)}")
    
    K = 100
    results = []
    
    grouped = combined_df.groupby(['drug_index', 'disease_index'])
    print(f"Number of unique drug-disease pairs: {len(grouped)}")
    
    for (drug_idx, dis_idx), group in grouped:
        seen = set()
        selected_paths = []
        
        unique_path_types = group['path_type'].unique()
        print(f"Drug {drug_idx} - Disease {dis_idx}: {len(group)} paths, Unique path types: {unique_path_types}")
        
        for _, row in group.iterrows():
            path_type = row['path_type']
            relations = path_type.split('-')
            relation_sequence = tuple([f"{relations[i]}-{relations[i+1]}" for i in range(len(relations)-1)])
            length = len(relation_sequence)
            
            if (relation_sequence, length) not in seen:
                seen.add((relation_sequence, length))
                
                node_types = relations
                nodes = []
                for n_type in node_types:
                    name_col = f"{n_type}_name"
                    if name_col in row:
                        nodes.append(row[name_col])
                    elif f"{n_type}1_name" in row and f"{n_type}2_name" in row:
                        nodes.append(row[f"{n_type}1_name"])
                        nodes.append(row[f"{n_type}2_name"])
                
                dis_cols = [c for c in group.columns if c.startswith("disease") and c.endswith("_index")]
                disease_index = row[dis_cols[0]] if dis_cols else None
                
                path_data = {
                    'drug_index': row['drug_index'],
                    'disease_index': disease_index,
                    'drug_name': row['drug_name'],
                    'disease_name': row[dis_cols[0].replace('_index', '_name')] if dis_cols else None,
                    'path_type': path_type,
                    'original_relation': row['original_relation'],
                    'nodes': str(nodes),
                    'score': 0.5
                }
                selected_paths.append(path_data)
                
                if len(selected_paths) >= K:
                    break
        
        results.extend(selected_paths)
        print(f"  Selected {len(selected_paths)} paths for Drug {drug_idx} - Disease {dis_idx}")

    output_df = pd.DataFrame(results)
    output_df = output_df[['drug_index', 'disease_index', 'drug_name', 'disease_name', 'path_type', 'original_relation', 'nodes', 'score']]
    output_df.to_csv("diverse_paths_output.csv", index=False)
    print(f"Total rows in output: {len(output_df)}")
    print("Output saved to 'diverse_paths_output.csv'")