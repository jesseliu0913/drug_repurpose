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

    
    

