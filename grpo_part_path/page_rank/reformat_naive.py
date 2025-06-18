import os
import numpy as np
import pandas as pd

file_path = "./train_grpo_naive.csv"
file_data = pd.read_csv(file_path)

refined_data = []

for index, row in file_data.iterrows():
    prefix = row['prefix'].split("\n")
    question = prefix[0]
    answer = prefix[-1]
    instruct = "directly answer me with $YES$ or $NO$"
    
    if "YES" in answer:
        answer = answer.replace("YES", "$YES$")
    elif "NO" in answer:
        answer = answer.replace("NO", "$NO$")
    
    refine_prefix = f"{question} {instruct}\n{answer}"
    
    row['prefix'] = refine_prefix
    refined_data.append(row)


refined_df = pd.DataFrame(refined_data)
refined_df.to_csv("train_grpo_naive_refined.csv", index=False)
