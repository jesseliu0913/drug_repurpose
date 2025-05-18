import os
import ast
import pandas as pd


negative_sample = pd.read_csv("negative_path.csv")
positive_sample = pd.read_csv("diverse_paths_output.csv")

def add_neg(negative_sample):
    for i, row in negative_sample.iterrows():
        answer = "NO"
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
            prompt = f"<phenotype>The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<phenotype>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-protein-drug":
            disease = nodes[0]
            gene = nodes[1]
            drug = nodes[2]
            prompt = f"<gene>The disease {disease} is associated with the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<gene>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        else:
            prefix = "NONE"
        
        negative_sample.at[i, 'prefix'] = prefix
    return negative_sample

# new token type: <degd>, <ddd>, <decgd>, <demgd>, <debgd>, <dppd>, <dpd>
def add_pos(positive_sample):
    for i, row in positive_sample.iterrows():
        answer = "YES"
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
            prompt = f"<phenotype>The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<phenotype>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-protein-drug":
            disease = nodes[0]
            gene = nodes[1]
            drug = nodes[2]
            prompt = f"<gene>The disease {disease} is associated with the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<gene>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        else:
            prefix = "NONE"
        
        positive_sample.at[i, 'prefix'] = prefix
    return positive_sample


negative_sample = add_neg(negative_sample)
positive_sample = add_pos(positive_sample)

negative_sample = negative_sample[negative_sample['prefix'] != "NONE"].copy()
positive_sample = positive_sample[positive_sample['prefix'] != "NONE"].copy()


neg_1000 = negative_sample.sample(n=1000, random_state=42)
pos_1000 = positive_sample.sample(n=1000, random_state=42)

# combined_sample = pd.concat([positive_sample, negative_sample], ignore_index=True)
# unique_pairs = combined_sample[['drug_index', 'disease_index']].drop_duplicates()
# sampled_pairs = unique_pairs.sample(n=500, random_state=42)
# final_sample = combined_sample.merge(sampled_pairs, on=['drug_index', 'disease_index'])
# final_sample = final_sample.reset_index(drop=True)

final_sample = pd.concat([pos_1000, neg_1000], ignore_index=True).sample(frac=1, random_state=42)
final_sample.to_csv("train_grpo.csv", index=False)