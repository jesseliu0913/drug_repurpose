import os
import glob
import pandas as pd
import networkx as nx
import numpy as np

test = pd.read_csv("/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv")
test_pairs = set(zip(test.drug_index, test.disease_index))

path_files = [f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/negative_path/*.csv") if f.endswith("_paths.csv")]
# path_files = [f for f in glob.glob("/playpen/jesse/drug_repurpose/grpo_path/path_data/*.csv") if f.endswith("_paths.csv")]

G = nx.DiGraph()
for fn in path_files:
    df = pd.read_csv(fn)
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    df = df[~df.apply(lambda r: any((r.drug_index, r[d]) in test_pairs for d in dis_cols), axis=1)]
    name_cols = [c for c in df.columns if c.endswith("_name")]
    for _, row in df.iterrows():
        nodes = [row[c] for c in name_cols]
        for u, v in zip(nodes, nodes[1:]):
            G.add_edge(u, v)

pr = nx.pagerank(G)

records = []
for fn in path_files:
    df = pd.read_csv(fn)
    dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    df = df[~df.apply(lambda r: any((r.drug_index, r[d]) in test_pairs for d in dis_cols), axis=1)]
    name_cols = [c for c in df.columns if c.endswith("_name")]
    for _, row in df.iterrows():
        nodes = [row[c] for c in name_cols]
        score = sum(pr.get(n, 0) for n in nodes)
        records.append({
            "drug_index":      row.drug_index,
            "disease_index":   row[dis_cols[-1]],
            "drug_name":       row.drug_name,
            "disease_name":    row[name_cols[0]],
            "path_type":       row.path_type,
            "original_relation": row.original_relation,
            "nodes":           nodes,
            "score":           score
        })

scored = pd.DataFrame(records)


top_k = 100
top_paths = (
    scored
    .sort_values("score", ascending=False)
    .groupby(["drug_index", "disease_index"])
    .head(top_k)
    .reset_index(drop=True)
)

def softmax_sample(gr):
    exp = np.exp(gr.score)
    return gr.sample(n=1, weights=exp / exp.sum())

sampled_paths = (
    scored
    .groupby(["drug_index", "disease_index"], group_keys=False)
    .apply(softmax_sample)
    .reset_index(drop=True)
)

top_paths.to_csv("train_paths_topk_negative.csv", index=False)
sampled_paths.to_csv("train_paths_sampled_negative.csv", index=False)
# top_paths.to_csv("train_paths_topk.csv", index=False)
# sampled_paths.to_csv("train_paths_sampled.csv", index=False)


