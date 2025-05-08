import os
import ast
import pandas as pd


negative_sample = pd.read_csv("train_paths_topk_negative.csv")
positive_sample = pd.read_csv("train_paths_topk.csv")

def add_neg(negative_sample):
    for i, row in negative_sample.iterrows():
        answer = "NO"
        nodes = row['nodes']
        nodes = ast.literal_eval(nodes)
        drug_name = row["drug_name"]
        disease_name = row["disease_name"]
        
        question = f"Is {disease_name} an indication for {drug_name}?"
        if row['path_type'] == "disease-exposure-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            gene = nodes[2]
            drug = nodes[3]
            prompt = f"<degd>The disease {disease} is associated with the exposure {exposure}, which in turn is linked to the gene {gene}. However, since the drug {drug} is not related to this gene, it may not be effective in treating {disease}.<degd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
            
        elif row['path_type'] == "disease-disease-drug":
            disease1 = nodes[0]
            disease2 = nodes[1]
            drug = nodes[2]
            prompt = f"<ddd>The disease {disease1} is linked to {disease2}, but since {disease2} is not associated with the drug {drug}, this suggests that {drug} may not be effective for treating {disease1}.<ddd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-exposure-cellcomp-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            cellcomp = nodes[2]
            gene = nodes[3]
            drug = nodes[4]
            prompt = f"<decgd>The disease {disease} is associated with the exposure {exposure}, which in turn affects the cell component {cellcomp}. This component is linked to the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<decgd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-exposure-molfunc-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            molfunc = nodes[2]
            gene = nodes[3]
            drug = nodes[4]
            prompt = f"<demgd>The disease {disease} is associated with the exposure {exposure}, which in turn affects the molecular function {molfunc}. This function is linked to the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<demgd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"

        elif row['path_type'] == "disease-exposure-bioprocess-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            bioprocess = nodes[2]
            gene = nodes[3]
            drug = nodes[4]
            prompt = f"<debgd>The disease {disease} is associated with the exposure {exposure}, which in turn affects the biological process {bioprocess}. This process is linked to the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<debgd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-phenotype-protein-drug":
            disease = nodes[0]
            phenotype = nodes[1]
            gene = nodes[2]
            drug = nodes[3]
            prompt = f"<dppd>The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<dppd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-protein-drug":
            disease = nodes[0]
            gene = nodes[1]
            drug = nodes[2]
            prompt = f"<dpd>The disease {disease} is associated with the gene {gene}, which is not targeted by the drug {drug}. These connections suggest that {drug} may not be effective in treating {disease}.<dpd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
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
        if row['path_type'] == "disease-exposure-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            gene = nodes[2]
            drug = nodes[3]
            prompt = f"<degd>The disease:{disease} is related to the exposure:{exposure}, which is related to the gene:{gene}, which is related to the drug:{drug}. This indicates that the drug:{drug} may be effective for the disease:{disease}.<degd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
            
        elif row['path_type'] == "disease-disease-drug":
            disease1 = nodes[0]
            disease2 = nodes[1]
            drug = nodes[2]
            prompt = f"<ddd>The disease:{disease1} is related to the disease:{disease2}, which is related to the drug:{drug}. This indicates that the drug:{drug} may be effective for the disease:{disease1}.<ddd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-exposure-cellcomp-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            cellcomp = nodes[2]
            gene = nodes[3]
            drug = nodes[4]
            prompt = f"<decgd>The disease {disease} is associated with the exposure {exposure}, which in turn affects the cell component {cellcomp}. This component is linked to the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<decgd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-exposure-molfunc-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            molfunc = nodes[2]
            gene = nodes[3]
            drug = nodes[4]
            prompt = f"<demgd>The disease {disease} is associated with the exposure {exposure}, which in turn affects the molecular function {molfunc}. This function is linked to the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<demgd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"

        elif row['path_type'] == "disease-exposure-bioprocess-gene-drug":
            disease = nodes[0]
            exposure = nodes[1]
            bioprocess = nodes[2]
            gene = nodes[3]
            drug = nodes[4]
            prompt = f"<debgd>The disease {disease} is associated with the exposure {exposure}, which in turn affects the biological process {bioprocess}. This process is linked to the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<debgd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-phenotype-protein-drug":
            disease = nodes[0]
            phenotype = nodes[1]
            gene = nodes[2]
            drug = nodes[3]
            prompt = f"<dppd>The disease {disease} is associated with the phenotype {phenotype}, which in turn affects the gene {gene}. This gene is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<dppd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        elif row['path_type'] == "disease-protein-drug":
            disease = nodes[0]
            gene = nodes[1]
            drug = nodes[2]
            prompt = f"<dpd>The disease {disease} is associated with the gene {gene}, which is targeted by the drug {drug}. These connections suggest that {drug} may be effective in treating {disease}.<dpd>"
            prefix = f"Question: {question}\nReasoning: {prompt}\nAnswer: {answer}"
        
        positive_sample.at[i, 'prefix'] = prefix
    return positive_sample


negative_sample = add_neg(negative_sample)
positive_sample = add_pos(positive_sample)

negative_sample_1000 = negative_sample.sample(n=1000, random_state=42)
positive_sample_1000 = positive_sample.sample(n=1000, random_state=42)
combined_df = pd.concat([negative_sample_1000, positive_sample_1000], ignore_index=True)
combined_df.to_csv('train_grpo.csv', index=False)
