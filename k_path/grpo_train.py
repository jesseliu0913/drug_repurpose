from __future__ import annotations
import argparse
import os
import json
import re
import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import wandb
from huggingface_hub import hf_hub_download
import argparse, json, os, os.path as osp, re
from pathlib import Path
from typing import List, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


HF_TOKEN = os.getenv("HF_API_TOKEN")
# ----------------------------- Helpers --------------------------------
def extract_answer(text: str) -> str | None:
    pattern = re.compile(r'Answer:\s*([^.\n]*)')
    m = pattern.search(text)
    return m.group(1).strip() if m else None

def is_lora_repo(repo: str) -> bool:
    try:
        hf_hub_download(repo, "adapter_config.json", token=HF_TOKEN)
        return True
    except Exception:
        return osp.exists(osp.join(repo, "adapter_config.json"))


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

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

@dataclass
class GRPOClippedConfig(GRPOConfig):
    clip_eps: float = field(
        default=0.2,
        metadata={"help": "Clip range for r_t in the E‑GRPO objective"}
    )

class GRPOClippedTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_eps: float = self.args.clip_eps

    def _compute_policy_loss(
        self,
        logprobs: torch.Tensor,            # (B, T)
        logprobs_old: torch.Tensor,        # (B, T)
        advantages: torch.Tensor,          # (B, T)
    ) -> torch.Tensor:
        ratio = (logprobs - logprobs_old).exp()          # r_t
        ratio_clipped = torch.clamp(ratio,
                                    1.0 - self.clip_eps,
                                    1.0 + self.clip_eps)
        surrogate = torch.minimum(ratio * advantages,
                                  ratio_clipped * advantages)
        return -surrogate.mean()

@torch.no_grad()
def build_kl_reward(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float = 0.05
):
    def reward_fn(
        prompts: List[str],
        completions: List[str],
        answers: List[str],
        **kwargs
    ) -> List[Optional[float]]:
        rewards: List[Optional[float]] = []
        for prompt, gen, gt in zip(prompts, completions, answers):
            pred = extract_answer(gen)
            if gt is None or pred is None:
                rewards.append(None)
                continue

            pred_tokens = pred.split()
            gt_tokens = gt.split()
            if pred_tokens and gt_tokens:
                common = set(pred_tokens) & set(gt_tokens)
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(gt_tokens)
                rl_r = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            else:
                rl_r = 0.0

            full = prompt + gen
            inputs = tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)
            logits_m = model(**inputs).logits
            logits_r = ref_model(**inputs).logits
            logp_m = F.log_softmax(logits_m, dim=-1)
            p_r = F.softmax(logits_r, dim=-1)
            kl = F.kl_div(logp_m, p_r, reduction="batchmean").item()
            rewards.append(rl_r - beta * kl)
        return rewards
    return reward_fn

# ----------------------------- Main -----------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GRPO training with KL penalty, clipping, W&B logging, and config saving"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="HF dataset path like Tassy24/K-Paths-inductive-reasoning-ddinter"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["ddinter", "drugbank", "pharmaDB"],
        required=True
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="./grpo_out")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    args = parser.parse_args()

    base_out = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    trial_dir = base_out / f"{timestamp}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # build W&B run name
    lora_flag = "lora" if args.use_lora else "no_lora"
    run_name = (
        f"grpo_{args.data_type}_"
        f"{Path(args.model_name).name}_"
        f"{lora_flag}_bs{args.batch_size}_iters{args.iterations}_gen{args.generations}_"
        f"lr{args.lr}_clip{args.clip_eps}_beta{args.beta}_{timestamp}"
    )

    # init W&B
    wandb.init(
        project="drug_repurposing",
        name=run_name,
        config={
            **vars(args),
            "run_name": run_name,
            "timestamp": timestamp
        }
    )

    with open(trial_dir / "config.json", "w") as cf:
        json.dump(vars(args), cf, indent=2)

    # ds = load_dataset(args.dataset, split="train")
    ds = load_dataset(args.dataset, split="train[:2000]")

    prompts, answers = [], []
    for ex in ds:
        if args.data_type == "ddinter":
            type_A = ex['drug1_name']
            type_B = ex['drug2_name']
            prompt = f"Question: What is the interaction severity between {type_A} and {type_B}?\nChoices:[Major, Moderate, Minor, No Interaction]\n\n"
        elif args.data_type == "drugbank":
            type_A = ex['drug1_name']
            type_B = ex['drug2_name']
            prompt = f"Question: What is the pharmacological interaction between {type_A} and {type_B}?"
        else:
            type_A = ex['drug_name']
            type_B = ex['disease_name']
            prompt = f"Question: What is the therapeutic relationship between {type_A} and {type_B}?\nChoices:[disease-modifying, palliates, non-indication]\n\n"
        prompts.append(prompt)
        answers.append(ex['label'])
    all_data = Dataset.from_dict({"prompt": prompts, "answers": answers})
    train_ds, eval_ds = all_data.train_test_split(test_size=0.1, seed=42).values()

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # model & optional LoRA
    if is_lora_repo(args.model_name):
        base_model = load_base_and_merge(args.model_name, tok)
        adapter_name = args.model_name 
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        adapter_name = None

    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]
        )
        model = get_peft_model(base_model, lora_cfg)
    else:
        model = base_model
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    # reward function
    kl_reward_fn = build_kl_reward(model, ref_model, tok, beta=args.beta)

    
    # ─────────────────────────────────────────────────────────────────────────────
    # GRPO trainer
    # ─────────────────────────────────────────────────────────────────────────────
    cfg = GRPOClippedConfig(
        output_dir=args.output_dir,
        # save_strategy="steps",
        # save_steps=10,
        num_iterations=args.iterations,
        num_generations=args.generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_prompt_length=512,
        clip_eps=args.clip_eps,
        max_completion_length=512,
        temperature=0.2,
        top_k=10,
        top_p=0.92,
        repetition_penalty=1.1,
        log_completions=True,
        logging_strategy="steps",
        logging_steps=20,
        lr_scheduler_type="cosine",
        report_to=["wandb"],               
        run_name=run_name,
    )
    from functools import partial


    trainer = GRPOClippedTrainer(
        model=model,
        processing_class=tok,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[kl_reward_fn],
    )

    trainer.train()

    # save artifacts
    trainer.save_model(str(trial_dir / "model"))
    tok.save_pretrained(trial_dir / "tokenizer")
    print(f"Artifacts saved to {trial_dir}")

if __name__ == "__main__":
    main()
