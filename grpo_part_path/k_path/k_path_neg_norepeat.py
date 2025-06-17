import os
import glob
import pandas as pd

# 1) Load your test set pairs and normalize to ints
test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test_pairs = set(
    zip(
        test.drug_index.astype(int),
        test.disease_index.astype(int)
    )
)
print(f"{len(test_pairs)} test‐pairs loaded")

# 2) Find your two path files
path_files = [
    f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/negative_path/*.csv")
    if os.path.basename(f) in [
        'disease_protein_drug_paths.csv',
        'disease_phenotype_protein_drug_paths.csv'
    ]
]
print("匹配到的 CSV 文件：", [os.path.basename(f) for f in path_files])

# 3) This set will hold all (drug, disease) pairs we've seen so far
seen_pairs = set(test_pairs)

# 4) We'll accumulate *all* formatted rows here, then write one output
all_rows = []
results = []

for file_path in path_files:
    df = pd.read_csv(file_path)

    # Identify which columns are “disease” indices
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]

    # 4a) Filter: drop any row where any (drug_index, that_dis_col) is already in seen_pairs
    def row_is_new(r):
        drug = int(r['drug_index'])
        # if you only mean the main disease_index column, you can replace dis_cols with ['disease_index']
        for dcol in dis_cols:
            pair = (drug, int(r[dcol]))
            if pair in seen_pairs:
                return False
        return True

    new_df = df[df.apply(row_is_new, axis=1)].copy()

    # 4b) Mark these pairs as seen so future files won’t emit duplicates
    for d_idx, dis_idx in zip(new_df.drug_index, new_df.disease_index):
        seen_pairs.add((int(d_idx), int(dis_idx)))

    # 4c) Format each remaining row
    for _, row in new_df.iterrows():
        # collect intermediate node names (everything ending in "_name", except drug_name & disease_name)
        nodes = [
            row[col]
            for col in df.columns
            if col.endswith("_name")
            and col not in ('drug_name','disease_name')
        ]
        all_rows.append({
            'drug_index': int(row['drug_index']),
            'disease_index': int(row['disease_index']),
            'drug_name': row['drug_name'],
            'disease_name': row['disease_name'],
            'path_type': row['path_type'],
            'original_relation': row['original_relation'],
            'nodes': str([row['disease_name']] + nodes + [row['drug_name']]),
            'score': 1
        })

    results.append({
        'file_name': os.path.basename(file_path),
        'original_rows': len(df),
        'filtered_out': len(df) - len(new_df),
        'emitted_rows': len(new_df)
    })
    print(f"Processed {os.path.basename(file_path)}: kept {len(new_df)} of {len(df)}")

# 5) Write out the combined negative paths CSV
out_df = pd.DataFrame(all_rows)
out_df.to_csv("./negative_path_norepeat.csv", index=False)
print(f" → Wrote {len(out_df)} total rows to negative_path_norepeat.csv")

# 6) And your summary
summary_df = pd.DataFrame(results)
summary_df.to_csv("filtering_results_summary_norepeat.csv", index=False)
print("Summary saved to filtering_results_summary_norepeat.csv")
