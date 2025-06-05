import glob, random, math
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

POS_DIR = "/playpen/jesse/drug_repurpose/grpo_path/path_data"
NEG_DIR = "/playpen/jesse/drug_repurpose/grpo_path/negative_path"
TEST_CSV = "/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv"

TARGET_PER_REL = 1000
TOP_K_PER_PAIR = 100
MAX_PT_PER_DRUG = 2
RNG_SEED = 2025

TOPK_ALL_CSV = "train_paths_topk.csv"
SAMPLED_ALL_CSV = "train_paths_sampled.csv"
BALANCED_CSV = "train_paths_balanced_diverse.csv"

rng = random.Random(RNG_SEED)

def read_paths(path_dir, label):
    frames = []
    for fp in glob.glob(f"{path_dir}/*_paths.csv"):
        df = pd.read_csv(fp)
        df["original_relation"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

test_pairs = set(pd.read_csv(TEST_CSV).pipe(lambda x: zip(x.drug_index, x.disease_index)))

def filter_test(df):
    dcols = [c for c in df.columns if c.startswith("disease") and c.endswith("_index")]
    mask = df.apply(lambda r: any((r.drug_index, r[d]) in test_pairs for d in dcols), axis=1)
    return df[~mask]

pos_df = filter_test(read_paths(POS_DIR, "indication"))
neg_df = filter_test(read_paths(NEG_DIR, "contraindication"))
all_df = pd.concat([pos_df, neg_df], ignore_index=True)

idx_cols = [c for c in all_df.columns if c.startswith("disease") and c.endswith("_index")]
name_cols_disease = [c for c in all_df.columns if c.startswith("disease") and c.endswith("_name")]

def sort_key(c):
    t = c[len("disease"):].split("_")[0]
    return int(t) if t.isdigit() else 0

idx_cols.sort(key=sort_key)
name_cols_disease.sort(key=sort_key)

token_col = {
    "disease":    "disease_name",
    "exposure":   "exposure_name",
    "bioprocess": "bioprocess_name",
    "gene":       "gene_name",
    "protein":    "protein_name",
    "drug":       "drug_name",
}

records = []
for _, row in all_df.iterrows():
    dis_val = next((row[c] for c in idx_cols if pd.notna(row[c])), None)
    dis_nm = next((row[c] for c in name_cols_disease if pd.notna(row[c])), None)
    if dis_val is None or dis_nm is None or pd.isna(row.drug_index):
        continue
    tokens = row.path_type.split("-")
    nodes = []
    dis_count = 0
    for tok in tokens:
        if tok == "disease":
            if dis_count == 0:
                nodes.append(dis_nm)
            else:
                col = f"disease{dis_count}_name"
                if col in row and pd.notna(row[col]):
                    nodes.append(row[col])
            dis_count += 1
        else:
            col = token_col.get(tok)
            if col and col in row and pd.notna(row[col]):
                nodes.append(row[col])
    records.append({
        "drug_index": int(row.drug_index),
        "disease_index": int(dis_val),
        "drug_name": row.drug_name,
        "disease_name": dis_nm,
        "path_type": row.path_type,
        "original_relation": row.original_relation,
        "nodes": str(nodes)
    })

paths_df = pd.DataFrame(records)

paths_df.groupby(["drug_index", "disease_index"], group_keys=False).head(TOP_K_PER_PAIR).to_csv(TOPK_ALL_CSV, index=False)
paths_df.groupby(["drug_index", "disease_index"], group_keys=False).apply(lambda g: g.sample(n=1, random_state=RNG_SEED)).reset_index(drop=True).to_csv(SAMPLED_ALL_CSV, index=False)

have_pos = set(pos_df.drug_index)
have_neg = set(neg_df.drug_index)
eligible_drugs = list(have_pos & have_neg)

bucket = defaultdict(lambda: {"indication": [], "contraindication": []})
for _, r in paths_df.groupby(["drug_index", "disease_index"], group_keys=False).apply(lambda g: g.sample(n=1, random_state=RNG_SEED)).iterrows():
    bucket[r.drug_index][r.original_relation].append(r)

pt_counts = paths_df.path_type.value_counts()
n_types = len(pt_counts)
total_needed = TARGET_PER_REL * 2
base_quota = math.ceil(total_needed / n_types)
quota = {pt: min(cnt, base_quota) for pt, cnt in pt_counts.items()}

cnt_global = Counter()
cnt_local = Counter()
balanced = []

for drug in rng.sample(eligible_drugs, len(eligible_drugs)):
    pos_list = bucket[drug]["indication"]
    neg_list = bucket[drug]["contraindication"]
    if not pos_list or not neg_list:
        continue
    pos = rng.choice(pos_list)
    neg_cand = [r for r in neg_list if r.path_type != pos.path_type] or neg_list
    neg = rng.choice(neg_cand)
    if cnt_global[pos.path_type] >= quota[pos.path_type] or cnt_global[neg.path_type] >= quota[neg.path_type]:
        continue
    if cnt_local[(drug, pos.path_type)] >= MAX_PT_PER_DRUG or cnt_local[(drug, neg.path_type)] >= MAX_PT_PER_DRUG:
        continue
    balanced.extend([pos, neg])
    cnt_global[pos.path_type] += 1
    cnt_global[neg.path_type] += 1
    cnt_local[(drug, pos.path_type)] += 1
    cnt_local[(drug, neg.path_type)] += 1
    if len(balanced) >= total_needed:
        break

pd.DataFrame(balanced).to_csv(BALANCED_CSV, index=False)
