import argparse, json, os, os.path as osp, re, sys, pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

SPECIAL_TOKENS = ["<degd>", "<ddd>", "<decgd>", "<demgd>", "<debgd>", "<dppd>", "<dpd>"]
HF_TOKEN = os.getenv("HF_API_TOKEN")

# ── CLI ────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model_name", default="JesseLiu/llama32-3b-cold")
p.add_argument("--train_csv", default="../grpo_path/train_grpo.csv")
p.add_argument("--output_dir", default="grpo-out")
p.add_argument("--per_device_train_batch_size", type=int, default=2)
p.add_argument("--gradient_accumulation_steps", type=int, default=4)
p.add_argument("--num_iterations", type=int, default=5)
p.add_argument("--learning_rate", type=float, default=1e-5)
p.add_argument("--num_generations", type=int, default=4)
p.add_argument("--use_lora", action="store_true")
p.add_argument("--lora_r", type=int, default=16)
p.add_argument("--lora_alpha", type=int, default=32)
p.add_argument("--lora_dropout", type=float, default=0.05)
args = p.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ── utils ─────────────────────────────────────────────────────────────────────
def is_lora_repo(repo: str) -> bool:
    try:
        hf_hub_download(repo, "adapter_config.json", token=HF_TOKEN); return True
    except Exception:
        return osp.exists(osp.join(repo, "adapter_config.json"))

def extract_question(text: str) -> str:
    m = re.search(r"Question:(.*?)Reasoning:", text, re.S)
    return (m.group(1) if m else text).strip()

def load_base_and_merge(adapter_repo: str, tokenizer):
    cfg = json.load(open(hf_hub_download(adapter_repo, "adapter_config.json", token=HF_TOKEN)))
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model_name_or_path"],
                                                device_map="auto", token=HF_TOKEN)
    base.resize_token_embeddings(len(tokenizer))
    merged = PeftModel.from_pretrained(base, adapter_repo, token=HF_TOKEN,
                                       is_trainable=True).merge_and_unload()
    return merged

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.train_csv)
prompts = [extract_question(t) for t in df["prefix"]]
train_ds, eval_ds = Dataset.from_dict({"prompt": prompts}).train_test_split(0.1, seed=42).values()

# ── tokenizer ────────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
tok.pad_token, tok.padding_side = tok.eos_token, "right"
tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

# ── model ─────────────────────────────────────────────────────────────────────
if is_lora_repo(args.model_name):
    model = load_base_and_merge(args.model_name, tok)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 device_map="auto", token=HF_TOKEN)
    model.resize_token_embeddings(len(tok))

if args.use_lora:
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj","k_proj","v_proj","o_proj",
                                          "gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora_cfg)

# ── reward ────────────────────────────────────────────────────────────────────
def reward_fn(prompts, completions, **kwargs):
    return [1.0 if sum(tok in c for tok in SPECIAL_TOKENS) == 1 else 0.0 for c in completions]

# ── GRPO config & trainer ────────────────────────────────────────────────────
cfg = GRPOConfig(
    output_dir=args.output_dir,
    num_iterations=args.num_iterations,
    num_generations=args.num_generations,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    max_prompt_length=256,
    max_completion_length=128,
    temperature=0.8, top_k=50, top_p=0.92, repetition_penalty=1.1,
    logging_strategy="steps", logging_steps=20,
    lr_scheduler_type="cosine", warmup_ratio=0.1)

trainer = GRPOTrainer(model=model, processing_class=tok, args=cfg,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      reward_funcs=reward_fn)
trainer.train()

# ── save ──────────────────────────────────────────────────────────────────────
model_path = osp.join(args.output_dir, "final_model")
model.save_pretrained(model_path)
tok.save_pretrained(model_path)
json.dump(vars(args), open(osp.join(args.output_dir, "training_config.json"), "w"), indent=2)
print(f"Done. Saved to {model_path}")
