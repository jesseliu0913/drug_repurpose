import os
import ast
import argparse
import jsonlines
from huggingface_hub import hf_hub_download
import argparse, json, os, os.path as osp, re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
fixed_seed = 42
HF_TOKEN = os.getenv("HF_API_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Experiments")
    parser.add_argument("--model_name", type=str, required=False, help="HuggingFace base model or adapter name")
    parser.add_argument("--task", type=str, default="", help="task")
    parser.add_argument("--shuffle_num", type=int, default=1, help="Number of times to shuffle the dataset")
    parser.add_argument("--adapter_path", type=str, default="", help="PEFT adapter PATH (optional)")
    parser.add_argument("--eval_type", type=str, default="test", help="Eval Type")
    parser.add_argument("--data_type", type=str, default="ddinter", help="DATA Type")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save JSONL outputs")
    return parser.parse_args()


def load_base_and_merge(adapter_repo: str, tokenizer):
    if os.path.isdir(adapter_repo):
        config_path = os.path.join(adapter_repo, "adapter_config.json")
    else:
        config_path = hf_hub_download(
            repo_id=adapter_repo,
            filename="adapter_config.json",
            repo_type="model",
            token=HF_TOKEN
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_name_or_path"],
        device_map="auto",
        use_auth_token=HF_TOKEN
    )

    # base.resize_token_embeddings(len(tokenizer))

    merged = (
        PeftModel
        .from_pretrained(
            base,
            adapter_repo,
            use_auth_token=HF_TOKEN,
            is_trainable=True
        )
        .merge_and_unload()
    )

    return merged

def load_model_and_tokenizer(task, model_name, adapter_name, hf_token):
    # Load tokenizer and model (with optional PEFT adapter)
    if adapter_name:
        if "grpo" not in task.lower():
            cfg = PeftConfig.from_pretrained(adapter_name, use_auth_token=hf_token)
            base_name = cfg.base_model_name_or_path or model_name
            tokenizer = AutoTokenizer.from_pretrained(adapter_name, use_auth_token=hf_token)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype="auto", use_auth_token=hf_token)
            model = PeftModel.from_pretrained(base, adapter_name, torch_dtype="auto")
        elif "grpo" in adapter_name.lower():
            tok_path = adapter_name.replace("model", "tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tok_path, use_auth_token=hf_token)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model = load_base_and_merge(adapter_name, tokenizer)
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
    hf_token = os.getenv("HF_API_TOKEN")
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"{args.data_type}.jsonl")
    if args.adapter_path and 'grpo' not in args.task.lower():
       task_folder = f"{args.task}-final{args.data_type}"
       adapter_name = f"{args.adapter_path}/{task_folder}"
    elif args.adapter_path and 'grpo' in args.task.lower():
       adapter_name = args.adapter_path
    else:
       adapter_name = None
    

    print(f"Loading model: {args.model_name}, adapter: {adapter_name}")
    model, tokenizer = load_model_and_tokenizer(args.task, args.model_name, adapter_name, hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load data
    assert args.data_type in ["ddinter", "drugbank", "pharmaDB"], "Invalid data type. Choose from ['ddinter', 'drugbank', 'pharmaDB']"
    path_data = load_dataset(f"Tassy24/K-Paths-inductive-reasoning-{args.data_type}")
    test_data = path_data["test"] if args.eval_type == "test" else path_data["train"]

    max_samples = 1000
    if len(test_data) > max_samples:
        test_data = test_data.shuffle(seed=fixed_seed).select(range(max_samples))


    with jsonlines.open(output_file, 'a') as writer:
        for i in range(len(test_data)):
            if args.data_type == "ddinter":
                type_A = test_data[i]['drug1_name']
                type_B = test_data[i]['drug2_name']
                prompt = f"Question: What is the interaction severity between {type_A} and {type_B}?\nChoices:[Major, Moderate, Minor, No Interaction]\n\n"
            elif args.data_type == "drugbank":
                type_A = test_data[i]['drug1_name']
                type_B = test_data[i]['drug2_name']
                prompt = f"Question: What is the pharmacological interaction between {type_A} and {type_B}?"
            elif args.data_type == "pharmaDB":
                type_A = test_data[i]['drug_name']
                type_B = test_data[i]['disease_name']
                prompt = f"Question: What is the therapeutic relationship between {type_A} and {type_B}?\nChoices:[Disease-modifying, Palliates, Non-indication]\n\n"

            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            answers = []
            for _ in range(max(1, args.shuffle_num)):
                out = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.2, top_k=50, top_p=0.9)
                txt = tokenizer.decode(out[0], skip_special_tokens=True)
                ans = txt.replace(prompt, "").strip()
                answers.append(ans)
            result = answers[0] if args.shuffle_num == 1 else answers

            writer.write({
                "A": type_A,
                "B": type_B,
                "prompt": prompt,
                "answer": result,
                "label": test_data[i]['label']
            })

    print("Done. Results written to", output_file)


if __name__ == "__main__":
    main()
