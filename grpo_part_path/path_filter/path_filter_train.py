import os
import glob
import pandas as pd
import numpy as np

test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test_pairs = set(zip(test.drug_index, test.disease_index))

# Combine both positive and negative path files
positive_path_files = [f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/path_data/*.csv") if f.endswith("_paths.csv")]
negative_path_files = [f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/negative_path/*.csv") if f.endswith("_paths.csv")]
path_files = positive_path_files + negative_path_files

# Process all path files and collect records
records = []
for fn in path_files:
    df = pd.read_csv(fn)
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    df = df[~df.apply(lambda r: any((r.drug_index, r[d]) in test_pairs for d in dis_cols), axis=1)]
    name_cols = [c for c in df.columns if c.endswith("_name")]
    
    for _, row in df.iterrows():
        nodes = [row[c] for c in name_cols]
        records.append({
            "drug_index":      row.drug_index,
            "disease_index":   row[dis_cols[-1]],
            "drug_name":       row.drug_name,
            "disease_name":    row[name_cols[0]],
            "path_type":       row.path_type,
            "original_relation": row.original_relation,
            "nodes":           nodes
        })

all_paths = pd.DataFrame(records)

# Separate indications and contraindications
indications = all_paths[all_paths['original_relation'] == 'indication'].copy()
contraindications = all_paths[all_paths['original_relation'] == 'contraindication'].copy()

# Get unique drugs
all_drugs = set(all_paths['drug_index'].unique())

# Function to balance sampling for each drug
def balance_drug_relations(drug_idx, ind_df, contra_df, target_per_relation=500):
    """Balance indications and contraindications for a single drug"""
    drug_ind = ind_df[ind_df['drug_index'] == drug_idx]
    drug_contra = contra_df[contra_df['drug_index'] == drug_idx]
    
    # Ensure at least 1 of each type
    if len(drug_ind) == 0 or len(drug_contra) == 0:
        return pd.DataFrame()  # Skip drugs without both types
    
    # Calculate how many to sample for each type
    total_available = len(drug_ind) + len(drug_contra)
    max_per_type = min(target_per_relation, total_available // 2)
    
    # Sample equal amounts from each type (at least 1 each)
    n_ind = min(max_per_type, len(drug_ind))
    n_contra = min(max_per_type, len(drug_contra))
    
    # Ensure at least 1 of each
    n_ind = max(1, n_ind)
    n_contra = max(1, n_contra)
    
    sampled_ind = drug_ind.sample(n=n_ind, random_state=42)
    sampled_contra = drug_contra.sample(n=n_contra, random_state=42)
    
    return pd.concat([sampled_ind, sampled_contra])

# Balance sampling for each drug
balanced_samples = []
for drug_idx in all_drugs:
    balanced_drug = balance_drug_relations(drug_idx, indications, contraindications)
    if not balanced_drug.empty:
        balanced_samples.append(balanced_drug)

if balanced_samples:
    balanced_paths = pd.concat(balanced_samples, ignore_index=True)
else:
    balanced_paths = pd.DataFrame()

# Now sample to get exactly 1000 indications and 1000 contraindications
final_indications = balanced_paths[balanced_paths['original_relation'] == 'indication']
final_contraindications = balanced_paths[balanced_paths['original_relation'] == 'contraindication']

# Sample exactly 1000 of each (or all available if less than 1000)
n_ind_sample = min(10000, len(final_indications))
n_contra_sample = min(10000, len(final_contraindications))

sampled_indications = final_indications.sample(n=n_ind_sample, random_state=42)
sampled_contraindications = final_contraindications.sample(n=n_contra_sample, random_state=42)

# Combine final samples
final_paths = pd.concat([sampled_indications, sampled_contraindications], ignore_index=True)

# Create top_k version (same as sampled in this case since we're doing balanced sampling)
top_paths = final_paths.copy()

# For sampled paths, we can do additional random sampling if needed
sampled_paths = final_paths.sample(frac=1, random_state=42).reset_index(drop=True)

# Save results
top_paths.to_csv("train_paths_topk.csv", index=False)
sampled_paths.to_csv("train_paths_sampled.csv", index=False)

# Print summary statistics
print(f"Total path files processed: {len(path_files)}")
print(f"  - Positive path files: {len(positive_path_files)}")
print(f"  - Negative path files: {len(negative_path_files)}")
print(f"Total final paths: {len(final_paths)}")
print(f"Indications: {len(final_paths[final_paths['original_relation'] == 'indication'])}")
print(f"Contraindications: {len(final_paths[final_paths['original_relation'] == 'contraindication'])}")
print(f"Unique drugs: {final_paths['drug_index'].nunique()}")

# Check balance per drug
drug_balance = final_paths.groupby(['drug_index', 'original_relation']).size().unstack(fill_value=0)
print("\nDrug balance summary:")
print(f"Drugs with both indications and contraindications: {len(drug_balance[(drug_balance['indication'] > 0) & (drug_balance['contraindication'] > 0)])}")
print(f"Total drugs: {len(drug_balance)}")