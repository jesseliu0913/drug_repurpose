import os
import glob
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict

test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test_pairs = set(zip(test.drug_index, test.disease_index))
print(len(test_pairs))
path_files = [f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/negative_path/*.csv")
              if os.path.basename(f) in ['disease_protein_drug_paths.csv', 'disease_phenotype_protein_drug_paths.csv']]

def process_file(file_path):
    df = pd.read_csv(file_path)

    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    filtered_df = df[~df.apply(lambda r: any((r.drug_index, r[d]) in test_pairs for d in dis_cols), axis=1)]
    
    # Create the new formatted dataframe
    formatted_rows = []
    for _, row in filtered_df.iterrows():
        # Extract node names in order
        nodes = []
        for col in df.columns:
            if col.endswith("_name") and not col.endswith("drug_name") and not col.endswith("disease_name"):
                nodes.append(row[col])
        
        # Create correctly formatted row
        formatted_row = {
            'drug_index': row['drug_index'],
            'disease_index': row['disease_index'],
            'drug_name': row['drug_name'],
            'disease_name': row['disease_name'],
            'path_type': row['path_type'],
            'original_relation': row['original_relation'],
            'nodes': str([row['disease_name']] + nodes + [row['drug_name']]),
            'score': 1
        }
        formatted_rows.append(formatted_row)
    
    formatted_df = pd.DataFrame(formatted_rows)
    
    output_path = "./negative_path.csv"
    formatted_df.to_csv(output_path, index=False)
    
    return {
        'file_name': os.path.basename(file_path),
        'original_rows': len(df),
        'filtered_rows': len(filtered_df),
        'formatted_rows': len(formatted_rows),
        'output_file': output_path
    }

results = []
for file_path in path_files:
    try:
        result = process_file(file_path)
        results.append(result)
        print(f"Processed: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        results.append({
            'file_name': os.path.basename(file_path),
            'error': str(e)
        })

results_df = pd.DataFrame(results)
results_df.to_csv("filtering_results_summary.csv", index=False)

print(f"Total files processed: {len(results)}")
print(f"Summary saved to: filtering_results_summary.csv")