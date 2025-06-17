import os
import glob
import pandas as pd

# 1) load & cast your test set pairs
test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test_pairs = set(zip(
    test.drug_index .astype(int),
    test.disease_index.astype(int)
))
print(f"Loaded {len(test_pairs)} test‐pairs")

# 2) we’ll track EVERYTHING we’ve already output here
seen_pairs = set(test_pairs)

# 3) find your path files
path_files = [
    f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/path_data/*.csv")
    if f.endswith("_paths.csv")
]
print(f"Found {len(path_files)} path files")

all_paths = []

for fn in path_files:
    df = pd.read_csv(fn)
    print(f"\nReading {fn}: {len(df)} rows")

    # cast drug_index + all disease_index cols
    df['drug_index'] = df['drug_index'].astype(int)
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    for dcol in dis_cols:
        df[dcol] = df[dcol].astype(int)

    # filter out any row whose pair is already in seen_pairs
    def is_new(r):
        d = r['drug_index']
        return all((d, r[dcol]) not in seen_pairs for dcol in dis_cols)

    before = len(df)
    df = df[df.apply(is_new, axis=1)].copy()
    after = len(df)
    print(f"  Filtered {before - after} rows; {after} remain")

    if after == 0:
        continue

    # mark these pairs as seen so no later file can re‐emit them
    for d_idx, dis_idx in zip(df['drug_index'], df[dis_cols[0]]):
        seen_pairs.add((d_idx, dis_idx))

    all_paths.append(df)

if not all_paths:
    print("No paths left after filtering all files.")
    exit()

# 4) combine and do your grouping logic
combined_df = pd.concat(all_paths, ignore_index=True)
print(f"\nTotal candidate rows: {len(combined_df)}")

K = 100
results = []

grouped = combined_df.groupby(['drug_index', 'disease_index'])
print(f"Unique drug–disease pairs to process: {len(grouped)}")

for (drug_idx, dis_idx), group in grouped:
    seen = set()
    selected = []

    print(f"\n→ Pair ({drug_idx}, {dis_idx}): {len(group)} rows")

    for _, row in group.iterrows():
        path_type = row['path_type']
        rels = path_type.split('-')
        seq = tuple(f"{rels[i]}-{rels[i+1]}" for i in range(len(rels)-1))
        length = len(seq)

        if (seq, length) in seen:
            continue
        seen.add((seq, length))

        # gather node names
        nodes = []
        for n_type in rels:
            col = f"{n_type}_name"
            if col in row:
                nodes.append(row[col])
            elif f"{n_type}1_name" in row and f"{n_type}2_name" in row:
                nodes.extend([row[f"{n_type}1_name"], row[f"{n_type}2_name"]])

        results.append({
            'drug_index':      drug_idx,
            'disease_index':   dis_idx,
            'drug_name':       row['drug_name'],
            'disease_name':    row.get(col.replace('_index','_name'), None),
            'path_type':       path_type,
            'original_relation': row['original_relation'],
            'nodes':           str(nodes),
            'score':           0.5
        })

        if len(selected) >= K:
            break

    print(f"  Selected {len(selected)} paths for this pair")

# 5) write out
output_df = pd.DataFrame(results)
cols = ['drug_index','disease_index','drug_name','disease_name','path_type','original_relation','nodes','score']
output_df = output_df[cols]
output_df.to_csv("diverse_paths_output_norepeat.csv", index=False)
print(f"\nWrote {len(output_df)} total rows to diverse_paths_output_norepeat.csv")
