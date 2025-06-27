import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def revise_path(path: str) -> None:
    new_path = path.replace("train_grpo_baseline", "train_grpo_baseline_revised")
    file_data = pd.read_csv(path)
    for index, row in file_data.iterrows():
        if row['prefix'] == 'None':
            continue
        question = row['prefix'].split("\n")[0]
        reasoning = row['prefix'].split("\n")[1].replace("Reasoning:", "").strip()
        answer = row['prefix'].split("\n")[-1]
        prefix = f"{question}\n Reasoning List:[{reasoning};]\n{answer}"
        file_data.at[index, 'prefix'] = prefix
    file_data.to_csv(new_path, index=False)

kpath_baseline = "/playpen/jesse/drug_repurpose/grpo_part_path/k_path/train_grpo_baseline.csv"
pagerank_baseline = "/playpen/jesse/drug_repurpose/grpo_part_path/page_rank/train_grpo_baseline.csv"

revise_path(kpath_baseline)
revise_path(pagerank_baseline)