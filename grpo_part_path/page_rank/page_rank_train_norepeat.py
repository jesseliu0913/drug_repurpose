import os
import glob
import pandas as pd
import networkx as nx
import numpy as np

# 1) load test pairs and seed seen_pairs
test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test['drug_index']    = test['drug_index'].astype(int)
test['disease_index'] = test['disease_index'].astype(int)
test_pairs = set(zip(test.drug_index, test.disease_index))

seen_pairs = set(test_pairs)

# 2) prepare
path_files = [
    f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/path_data/*.csv")
    if f.endswith("_paths.csv")
]

G = nx.DiGraph()
records = []

# 3) single pass: build graph & collect records, skipping seen pairs
for fn in path_files:
    df = pd.read_csv(fn)
    # cast indices
    df['drug_index'] = df['drug_index'].astype(int)
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    for dcol in dis_cols:
        df[dcol] = df[dcol].astype(int)

    name_cols = [c for c in df.columns if c.endswith("_name")]

    print(f"Reading {os.path.basename(fn)}: {len(df)} rows")

    for _, row in df.iterrows():
        pair = (row['drug_index'], row[dis_cols[-1]])
        if pair in seen_pairs:
            continue

        # mark it seen
        seen_pairs.add(pair)

        # build edges
        nodes = [row[c] for c in name_cols]
        for u, v in zip(nodes, nodes[1:]):
            G.add_edge(u, v)

        # stash for scoring
        records.append({
            "drug_index":        row['drug_index'],
            "disease_index":     row[dis_cols[-1]],
            "drug_name":         row['drug_name'],
            "disease_name":      row[name_cols[0]],        # first disease_name col
            "path_type":         row['path_type'],
            "original_relation": row['original_relation'],
            "nodes":             nodes,
            # score to fill in later
            "score":             None
        })

print(f"\nBuilt graph on {len(G.nodes)} nodes and collected {len(records)} unique paths")

# 4) PageRank & assign scores
pr = nx.pagerank(G)
for rec in records:
    rec['score'] = sum(pr.get(n, 0.0) for n in rec['nodes'])

scored = pd.DataFrame(records)

# 5) top-K by score
top_k = 100
top_paths = (
    scored
    .sort_values("score", ascending=False)
    .groupby(["drug_index", "disease_index"], as_index=False)
    .head(top_k)
)
top_paths.to_csv("train_paths_topk_pos_norepeat.csv", index=False)

# 6) softmax sampling
def softmax_sample(gr):
    exp = np.exp(gr.score)
    return gr.sample(n=1, weights=exp/exp.sum())

sampled = (
    scored
    .groupby(["drug_index", "disease_index"], group_keys=False)
    .apply(softmax_sample)
    .reset_index(drop=True)
)
sampled.to_csv("train_paths_sampled_pos_norepeat.csv", index=False)

print("Done: wrote train_paths_topk_norepeat.csv and train_paths_sampled_norepeat.csv")
