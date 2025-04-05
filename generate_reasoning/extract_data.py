from pathlib import Path
import json
import random
random.seed(42)

ddgene_path = Path("/playpen/jesse/drug_repurpose/split_data/data/ddgene.jsonl")
ddphenotype_path = Path("/playpen/jesse/drug_repurpose/split_data/data/ddgene.jsonl")
combined_path = Path("./data/combined.jsonl")

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

ddgene_data = read_file(ddgene_path)
ddphenotype_data = read_file(ddphenotype_path)

combined_data = ddgene_data + ddphenotype_data
random.shuffle(combined_data)
combined_data = combined_data[:2000]

with open(combined_path, "w", encoding="utf-8") as f:
    for item in combined_data:
        f.write(json.dumps(item) + "\n")
