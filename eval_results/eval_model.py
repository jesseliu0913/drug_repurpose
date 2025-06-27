import os
import ast
import argparse
import jsonlines

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Experiments")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace base model or adapter name")
    parser.add_argument("--adapter_name", type=str, default="", help="PEFT adapter name (optional)")
    parser.add_argument("--eval_type", type=str, default="test", help="Eval Type")
    parser.add_argument("--input_file", type=str, default="../split_data/data_analysis/test_data_new.csv", help="CSV file with test/train data")
    parser.add_argument("--nodes_file", type=str, default="../PrimeKG/nodes.csv", help="CSV file with node index-to-name mapping")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save JSONL outputs")
    parser.add_argument("--prompt_type", type=str, choices=["raw", "cot", "fcot", "phenotype", "gene", "fraw", "raw3"], required=True, help="Prompt style to use")
    parser.add_argument("--shuffle_num", type=int, default=1, help="Number of samples per query for ensembling")
    return parser.parse_args()


def load_model_and_tokenizer(model_name, adapter_name, hf_token):
    # Load tokenizer and model (with optional PEFT adapter)
    if adapter_name:
        if "grpo" not in adapter_name.lower():
            cfg = PeftConfig.from_pretrained(adapter_name, use_auth_token=hf_token)
            base_name = cfg.base_model_name_or_path or model_name
            tokenizer = AutoTokenizer.from_pretrained(adapter_name, use_auth_token=hf_token)
            base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype="auto", use_auth_token=hf_token)
            model = PeftModel.from_pretrained(base, adapter_name, torch_dtype="auto")
        else:
            tokenizer = AutoTokenizer.from_pretrained(adapter_name, use_auth_token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(adapter_name, torch_dtype="auto", use_auth_token=hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", use_auth_token=hf_token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def main():
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"{args.prompt_type}.jsonl")

    print(f"Loading model: {args.model_name}, adapter: {args.adapter_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_name, hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load data
    data = pd.read_csv(args.input_file)
    nodes = pd.read_csv(args.nodes_file)
    print(f"Currently eval type is {args.eval_type}")
    print(f"Data file: {args.input_file}, {len(data)} samples")

    existing = set()
    if os.path.exists(output_file):
        with jsonlines.open(output_file, 'r') as r:
            for obj in r:
                existing.add((obj.get("drug_name"), obj.get("disease_name")))

    two_shot = """
        Question: Is Fosinopril an indication for hypertensive disorder?
        REASONING: Fosinopril is indicated for hypertensive disorders because it functions as an angiotensin-converting enzyme (ACE) inhibitor, which blocks the conversion of angiotensin I to angiotensin II—a potent vasoconstrictor. By reducing angiotensin II levels, Fosinopril promotes vasodilation, decreases peripheral vascular resistance, and ultimately lowers blood pressure. This mechanism directly addresses the pathophysiology of hypertension, making Fosinopril an effective and commonly prescribed medication for managing high blood pressure and reducing the risk of associated cardiovascular complications.
        ANSWER:$YES$
        Question: Is Rotigotine an indication for hypertensive disorder?
        REASONING: Rotigotine is a dopamine agonist primarily used to treat Parkinson’s disease and restless legs syndrome (RLS). It works by stimulating dopamine receptors in the brain to help manage motor symptoms. While it may have some effects on blood pressure as a side effect (e.g., causing orthostatic hypotension), it is not approved or used as a treatment for hypertension or other hypertensive disorders.
        ANSWER:$NO$
        """
    raw_shot = (
        "Question: Is Fosinopril an indication for hypertensive disorder?\nANSWER:$YES$\n"
        "Question: Is Rotigotine an indication for hypertensive disorder?\nANSWER:$NO$"
    )

    with jsonlines.open(output_file, 'a') as writer:
        for _, row in data.iterrows():
            drug = row.drug_name
            disease = row.disease_name
            label = row.get("relation", row.get("original_relation"))
            if (drug, disease) in existing:
                continue

            phenos, genes = [], []
            if "related_phenotypes" in data.columns and pd.notna(row.related_phenotypes):
                for idx in ast.literal_eval(row.related_phenotypes):
                    phenos.append(nodes.loc[idx, 'node_name'])
                phenos = phenos[:10]
            if "related_proteins" in data.columns and pd.notna(row.related_proteins):
                for idx in ast.literal_eval(row.related_proteins):
                    genes.append(nodes.loc[idx, 'node_name'])
                genes = genes[:10]

            question = f"Is {disease} an indication for {drug}?"
            if args.prompt_type == "phenotype":
                prefix = f"{disease} has phenotypes: {phenos}\n"
                prompt = f"Question: {prefix}{question} DIRECTLY ANSWER $YES$ or $NO$\nANSWER:"
            elif args.prompt_type == "gene":
                prefix = f"{disease} associated genes: {genes}\n"
                prompt = f"Question: {prefix}{question} DIRECTLY ANSWER $YES$ or $NO$\nANSWER:"
            elif args.prompt_type == "cot":
                prompt = f"Question: {question} let's think step by step then answer $YES$ or $NO$\nREASONING:\nANSWER:"
            elif args.prompt_type == "fcot":
                prompt = f"{two_shot}\nQuestion: {question}\nANSWER:"
            elif args.prompt_type == "fraw":
                prompt = f"{raw_shot}\nQuestion: {question}\nANSWER:"
            elif args.prompt_type == "raw3":
                prompt = f"{raw_shot}\nQuestion: {question} answer $YES$, $NO$ or $Not Sure$\nANSWER:"
            else:  
                prompt = f"Question: {question} DIRECTLY ANSWER $YES$ or $NO$\nANSWER:"

            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            answers = []
            for _ in range(max(1, args.shuffle_num)):
                out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.2, top_k=50, top_p=0.9)
                txt = tokenizer.decode(out[0], skip_special_tokens=True)
                ans = txt.replace(prompt, "").strip()
                answers.append(ans)
            result = answers[0] if args.shuffle_num == 1 else answers

            writer.write({
                "drug_name": drug,
                "disease_name": disease,
                "prompt": prompt,
                "answer": result,
                "label": label
            })

    print("Done. Results written to", output_file)


if __name__ == "__main__":
    main()
