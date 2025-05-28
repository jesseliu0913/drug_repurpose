from __future__ import annotations
import argparse, json, os, os.path as osp, re
import pandas as pd
from typing import List, Optional
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import os
os.environ["WANDB_DISABLED"] = "true"
# ─────────────────────────────────────────────────────────────────────────────
# constants & env
# ─────────────────────────────────────────────────────────────────────────────
SPECIAL_TOKENS = [
    "<degd>", "<ddd>", "<decgd>", "<demgd>", "<debgd>", "<dppd>", "<dpd>"
]
ANS_TAG_RE     = re.compile(r"Answer\s*:?\s*(YES|NO)", re.I)
COMPLETION_RE  = re.compile(r"\b(YES|NO)\b", re.I)

load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="JesseLiu/llama32-3b-cold")
parser.add_argument("--train_csv",
                    default="/playpen/hongxuan/drug_repurpose/grpo_path/page_rank/train_grpo.csv")
parser.add_argument("--output_dir", default="grpo-out")
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--num_iterations", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_generations", type=int, default=4)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_lora_repo(repo: str) -> bool:
    try:
        hf_hub_download(repo, "adapter_config.json", token=HF_TOKEN)
        return True
    except Exception:
        return osp.exists(osp.join(repo, "adapter_config.json"))


def extract_question(text: str) -> str:
    m = re.search(r"Question:(.*?)Reasoning:", text, re.S)
    return (m.group(1) if m else text).strip()


def load_base_and_merge(adapter_repo: str, tokenizer):
    cfg = json.load(open(hf_hub_download(adapter_repo,
                                         "adapter_config.json",
                                         token=HF_TOKEN)))
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_name_or_path"], device_map="auto", token=HF_TOKEN)
    base.resize_token_embeddings(len(tokenizer))
    merged = PeftModel.from_pretrained(
        base, adapter_repo, token=HF_TOKEN,
        is_trainable=True).merge_and_unload()
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# model_name-aware utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_model_name(model_name: str) -> str:
    n = model_name.lower()
    if "qwen" in n:
        return "qwen"
    if "llama" in n:
        return "llama"
    return "other"


class TokenDecoderWrapper:
    """Keep special tokens; handle Qwen extra kwarg."""
    def __init__(self, tokenizer, model_name: str):
        self.tok, self.model_name = tokenizer, model_name

    def batch_decode(self, seqs, **kw):
        kw.pop("skip_special_tokens", None)
        kw.setdefault("skip_special_tokens", False)
        kw.setdefault("clean_up_tokenization_spaces", False)
        if self.model_name == "qwen":
            kw.setdefault("spaces_between_special_tokens", False)
        return self.tok.batch_decode(seqs, **kw)

    def decode(self, seq, **kw):
        kw.setdefault("skip_special_tokens", False)
        kw.setdefault("clean_up_tokenization_spaces", False)
        if self.model_name == "qwen":
            kw.setdefault("spaces_between_special_tokens", False)
        return self.tok.decode(seq, **kw)

    def __call__(self, *a, **kw):
        return self.tok(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.tok, name)


def get_lora_targets(model_name: str) -> List[str]:
    fam = get_model_name(model_name)
    if fam == "qwen":
        return ["wqkv", "wo", "w1", "w2", "w3", "embed_tokens"]
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "embed_tokens"
    ]

# ─────────────────────────────────────────────────────────────────────────────
# data  — prompt + ground-truth answer
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.train_csv).iloc[:2000]

raw_prefixes = df["prefix"].tolist()
prompts      = [extract_question(t) for t in raw_prefixes]

answers_gt = [
    (m.group(1).upper() if (m := ANS_TAG_RE.search(t)) else None)
    for t in raw_prefixes
]

train_ds, eval_ds = Dataset.from_dict(
    {"prompt": prompts, "answer": answers_gt}
).train_test_split(0.1, seed=42).values()

# ─────────────────────────────────────────────────────────────────────────────
# tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tok_raw = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
tok_raw.pad_token = tok_raw.eos_token
tok_raw.padding_side = "left"
# tok_raw.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

model_name = get_model_name(args.model_name)
tok = TokenDecoderWrapper(tok_raw, model_name)

# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
if is_lora_repo(args.model_name):
    model = load_base_and_merge(args.model_name, tok_raw)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", token=HF_TOKEN)
    model.resize_token_embeddings(len(tok_raw))

if args.use_lora:
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=get_lora_targets(args.model_name)
    )
    model = get_peft_model(model, lora_cfg)

# ─────────────────────────────────────────────────────────────────────────────
# reward helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract(text: str, regexes) -> Optional[str]:
    for rgx in regexes:
        if (m := rgx.search(text)):
            return m.group(1).upper()
    return None


def extract_pred(completion: str) -> Optional[str]:
    """YES/NO prediction from completion."""
    return _extract(completion, [ANS_TAG_RE, COMPLETION_RE])

# ---------------- rewards ----------------
def format_reward(prompts, completions, **kw):
    return [1.0 if sum(t in c for t in SPECIAL_TOKENS) == 1 else 0.0
            for c in completions]


def task_reward(prompts, completions, answer, **kw):
    out = []
    for gt, comp in zip(answer, completions):
        pred = extract_pred(comp)
        if gt is None or pred is None:
            out.append(None)                # ignored by GRPO
        else:
            out.append(1.0 if gt == pred else 0.0)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# GRPO trainer
# ─────────────────────────────────────────────────────────────────────────────
cfg = GRPOConfig(
    output_dir=args.output_dir,
    save_strategy="no",
    num_iterations=args.num_iterations,
    num_generations=args.num_generations,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    max_prompt_length=256,
    max_completion_length=128,
    temperature=0.8,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.1,
    log_completions=True,
    logging_strategy="steps",
    logging_steps=20,
    lr_scheduler_type="cosine",
    report_to=["wandb"],               
    run_name=f"grpo-drug-repurpose-{args.model_name}",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    reward_funcs=[task_reward],
)

trainer.train()

# ─────────────────────────────────────────────────────────────────────────────
# save
# ─────────────────────────────────────────────────────────────────────────────
model_path = osp.join(args.output_dir, "final_model")
model.save_pretrained(model_path)
tok_raw.save_pretrained(model_path)
json.dump(vars(args), open(osp.join(args.output_dir, "training_config.json"), "w"), indent=2)
print(f"Done. Saved to {model_path}")
