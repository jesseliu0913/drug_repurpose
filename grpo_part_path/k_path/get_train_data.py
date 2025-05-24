import os
import ast
import pandas as pd


negative_sample = pd.read_csv("negative_path.csv")
positive_sample = pd.read_csv("diverse_paths_output.csv")
combined_sample = pd.concat([positive_sample, negative_sample], ignore_index=True)

# def add_neg(negative_sample):
#     for i, row in negative_sample.iterrows():
#         answer = "NO"
#         nodes = row['nodes']
#         nodes = ast.literal_eval(nodes)
#         drug_name = row["drug_name"]
#         disease_name = row["disease_name"]
        
#         question = f"Is {disease_name} an indication for {drug_name}?"
#         if row['path_type'] == "disease-phenotype-protein-drug":
#             disease = nodes[0]
#             phenotype = nodes[1]
#             gene = nodes[2]
#             drug = nodes[3]
#             # prompt = f"<phenotype>The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<phenotype>"
#             prompt = f"The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}."
#             # prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
#             prefix = f"Question: {question}\nAnswer: {answer}"
        
#         elif row['path_type'] == "disease-protein-drug":
#             disease = nodes[0]
#             gene = nodes[1]
#             drug = nodes[2]
#             # prompt = f"<gene>The disease {disease} is associated with the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<gene>"
#             prompt = f"The disease {disease} is associated with the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}."
#             # prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
#             prefix = f"Question: {question}\nAnswer: {answer}"
        
#         else:
#             prefix = "NONE"
        
#         negative_sample.at[i, 'prefix'] = prefix
#     return negative_sample

# new token type: <degd>, <ddd>, <decgd>, <demgd>, <debgd>, <dppd>, <dpd>
def add_pos(combined_sample):
    for i, row in combined_sample.iterrows():
        answer = "YES" if row['original_relation'] == 'indication' else "NO"
        nodes = row['nodes']
        nodes = ast.literal_eval(nodes)
        drug_name = row["drug_name"]
        disease_name = row["disease_name"]
        
        question = f"Is {disease_name} an indication for {drug_name}?"
        
        if row['path_type'] == "disease-phenotype-protein-drug":
            disease = nodes[0]
            phenotype = nodes[1]
            gene = nodes[2]
            drug = nodes[3]
            # prompt = f"<phenotype>The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<phenotype>"
            prompt = f"The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}."
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
            # prefix = f"Question: {question}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-protein-drug":
            disease = nodes[0]
            gene = nodes[1]
            drug = nodes[2]
            # prompt = f"<gene>The disease {disease} is associated with the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<gene>"
            prompt = f"The disease {disease} is associated with the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}."
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
            # prefix = f"Question: {question}\nAnswer: {answer}"
        else:
            prefix = "NONE"
        
        positive_sample.at[i, 'prefix'] = prefix
    return positive_sample


# negative_sample = add_neg(negative_sample)
# positive_sample = add_pos(positive_sample)
combined_sample = add_pos(combined_sample)

# # negative_sample = negative_sample[negative_sample['prefix'] != "NONE"].copy()
# # positive_sample = positive_sample[positive_sample['prefix'] != "NONE"].copy()
combined_sample = combined_sample[combined_sample['prefix'] != "NONE"].copy()

# # neg_1000 = negative_sample.sample(n=1000, random_state=42)
# # pos_1000 = positive_sample.sample(n=1000, random_state=42)
# combined_sample = combined_sample.sample(n=2000, random_state=42)

# # pos_1000.to_csv("train_grpo_pos_naive.csv", index=False)
# # neg_1000.to_csv("train_grpo_neg_naive.csv", index=False)
# # final = pd.concat([pos_1000, neg_1000], ignore_index=True).sample(frac=1, random_state=42)
# combined_sample.to_csv("train_grpo_naive.csv", index=False)

yes_samples = combined_sample[combined_sample['original_relation'] == 'indication'].sample(n=1000, random_state=42)
no_samples = combined_sample[combined_sample['original_relation'] == 'contraindication'].sample(n=1000, random_state=42)

# Combine and shuffle
final_sample = pd.concat([yes_samples, no_samples], ignore_index=True).sample(frac=1, random_state=42)

# Save to CSV
final_sample.to_csv("train_grpo_baseline.csv", index=False)