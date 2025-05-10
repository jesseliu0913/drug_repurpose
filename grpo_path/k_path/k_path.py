#!/usr/bin/env python3
import os
import glob
import pandas as pd
import networkx as nx

def load_test_pairs(path):
    df = pd.read_csv(path)
    return set(zip(df.drug_index, df.disease_index))

def load_candidate_paths(path_dir, test_pairs):
    files = glob.glob(os.path.join(path_dir, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dis_cols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
        mask = df.apply(lambda r: any((r.drug_index, r[d]) in test_pairs for d in dis_cols), axis=1)
        df2 = df[~mask]
        if not df2.empty:
            dfs.append(df2)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def extract_diverse_k_paths(df, K):
    if df.empty:
        return pd.DataFrame()
    df = df[df['path_type'].notna()]
    name_cols = [c for c in df.columns if c.endswith("_name")]
    df['hop'] = df['path_type'].str.count("-") + 1
    df = df.sort_values('hop')
    gb = df.groupby(["drug_index", "disease_index", "drug_name", "disease_name"], as_index=False)
    rec = []
    for _, g in gb:
        seen = set()
        for _, r in g.iterrows():
            seq = tuple(r.path_type.split("-"))
            key = (seq, len(seq))
            if key in seen:
                continue
            seen.add(key)
            nodes = [x for x in (r[c] for c in name_cols) if pd.notnull(x)]
            rec.append({
                "drug_index":        r.drug_index,
                "disease_index":     r.disease_index,
                "drug_name":         r.drug_name,
                "disease_name":      r.disease_name,
                "path_type":         r.path_type,
                "original_relation": r.original_relation,
                "nodes":             nodes
            })
            if len(seen) >= K:
                break
    return pd.DataFrame(rec)

def build_graph(df):
    G = nx.DiGraph()
    for _, r in df.iterrows():
        name_cols = [c for c in df.columns if c.endswith("_name")]
        nodes = [x for x in (r[c] for c in name_cols) if pd.notnull(x)]
        for u, v in zip(nodes, nodes[1:]):
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=1)
    return G

def main():
    TEST_DATA = "/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv"
    PATH_DIR  = "../path_data"
    OUTPUT    = "train_paths_diverse_k_positive.csv"
    TOP_K     = 10
    test_pairs = load_test_pairs(TEST_DATA)
    df_cand    = load_candidate_paths(PATH_DIR, test_pairs)
    df_div     = extract_diverse_k_paths(df_cand, TOP_K)
    G = build_graph(df_cand)
    pr = nx.pagerank(G)
    df_div['score'] = df_div['nodes'].apply(lambda ns: sum(pr.get(n, 0) for n in ns))
    cols = ["drug_index","disease_index","drug_name","disease_name","path_type","original_relation","nodes","score"]
    df_div.to_csv(OUTPUT, index=False, columns=cols)

if __name__ == "__main__":
    main()
