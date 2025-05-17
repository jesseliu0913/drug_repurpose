import argparse, json, os, os.path as osp, re, sys, pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
from inspect import signature
from transformers import PreTrainedTokenizerBase

SPECIAL_TOKENS = ["<degd>", "<ddd>", "<decgd>", "<demgd>", "<debgd>", "<dppd>", "<dpd>"]
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

# ── CLI ────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model_name", default="JesseLiu/llama32-3b-cold")
p.add_argument("--train_csv", default="/playpen/hongxuan/drug_repurpose/grpo_path/page_rank/train_grpo.csv")
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
df = pd.read_csv(args.train_csv).iloc[:2000, :]
prompts = [extract_question(t) for t in df["prefix"]]
train_ds, eval_ds = Dataset.from_dict({"prompt": prompts}).train_test_split(0.1, seed=42).values()


tok = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
tok.pad_token, tok.padding_side = tok.eos_token, "left"
tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

print("Special tokens in tokenizer:", tok.special_tokens_map_extended)

for t in SPECIAL_TOKENS:
    tid = tok.convert_tokens_to_ids(t)
    print(f"{t} → id {tid}")

for t in SPECIAL_TOKENS:
    print(f"tokenize({t}) →", tok.tokenize(t))


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


class TokenDecoderWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def batch_decode(self, sequences, **kwargs):
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=False,
            **{k:v for k,v in kwargs.items() if k != 'skip_special_tokens'}
        )
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


# class TokenDecoderWrapper:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self._accepts_skip = "skip_special_tokens" in signature(tokenizer.batch_decode).parameters

#     def batch_decode(self, seqs, **kw):
#         if self._accepts_skip:
#             kw.setdefault("skip_special_tokens", False)
#             return self.tokenizer.batch_decode(seqs, **kw)
#         else:
#             return [self.tokenizer.decode(s, clean_up_tokenization_spaces=False, **kw)
#                     for s in seqs]

#     def decode(self, seq, **kw):
#         kw.setdefault("skip_special_tokens", False)
#         return self.tokenizer.decode(seq, **kw)
#     def __len__(self):
#         return len(self.tokenizer)

#     def __call__(self, *args, **kwargs):
#         return self.tokenizer(*args, **kwargs)

#     def __getattr__(self, name):
#         return getattr(self.tokenizer, name)

# class TokenDecoderWrapper(PreTrainedTokenizerBase):
#     def __init__(self, tokenizer):
#         # 复用底层一切属性 / 方法
#         self._wrapped = tokenizer
#         self.__dict__.update(tokenizer.__dict__)

#     # -------- 关键：批量解码 --------
#     def batch_decode(self, seqs, **kw):
#         kw["skip_special_tokens"] = False          # 覆盖上游设置
#         kw.setdefault("clean_up_tokenization_spaces", False)
#         return self._wrapped.batch_decode(seqs, **kw)

#     # -------- 关键：单条解码 --------
#     def decode(self, seq, **kw):                   # ★ 新增
#         kw["skip_special_tokens"] = False
#         kw.setdefault("clean_up_tokenization_spaces", False)
#         return self._wrapped.decode(seq, **kw)

#     # 透明转发
#     def __call__(self, *a, **kw):
#         return self._wrapped(*a, **kw)

#     def __getattr__(self, name):
#         return getattr(self._wrapped, name)

#     def __len__(self):
#         return len(self._wrapped)


# group size + learning rate 

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
    log_completions=True,
    logging_strategy="steps", logging_steps=20,
    lr_scheduler_type="cosine")# , warmup_ratio=0.1)

trainer = GRPOTrainer(model=model, processing_class=TokenDecoderWrapper(tok), args=cfg,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      reward_funcs=reward_fn)
trainer.train()

# ── save ──────────────────────────────────────────────────────────────────────
model_path = osp.join(args.output_dir, "final_model")
model.save_pretrained(model_path)
tok.save_pretrained(model_path)
json.dump(vars(args), open(osp.join(args.output_dir, "training_config.json"), "w"), indent=2)
print(f"Done. Saved to {model_path}")


# import argparse, json, os, os.path as osp, re, sys, pandas as pd
# from datasets import Dataset
# from huggingface_hub import hf_hub_download
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel, LoraConfig, get_peft_model
# from trl import GRPOConfig, GRPOTrainer
# from dotenv import load_dotenv
# from transformers import PreTrainedTokenizerBase

# SPECIAL_TOKENS = ["<degd>", "<ddd>", "<decgd>", "<demgd>", "<debgd>", "<dppd>", "<dpd>"]
# load_dotenv()
# HF_TOKEN = os.getenv("HF_API_TOKEN")

# # ── CLI ────────────────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser()
# p.add_argument("--model_name", default="JesseLiu/llama32-3b-cold")
# p.add_argument("--train_csv", default="/playpen/hongxuan/drug_repurpose/grpo_path/page_rank/train_grpo.csv")
# p.add_argument("--output_dir", default="grpo-out")
# p.add_argument("--per_device_train_batch_size", type=int, default=2)
# p.add_argument("--gradient_accumulation_steps", type=int, default=4)
# p.add_argument("--num_iterations", type=int, default=5)
# p.add_argument("--learning_rate", type=float, default=1e-5)
# p.add_argument("--num_generations", type=int, default=4)
# p.add_argument("--use_lora", action="store_true")
# p.add_argument("--lora_r", type=int, default=16)
# p.add_argument("--lora_alpha", type=int, default=32)
# p.add_argument("--lora_dropout", type=float, default=0.05)
# args = p.parse_args()
# os.makedirs(args.output_dir, exist_ok=True)

# # ── utils ─────────────────────────────────────────────────────────────────────
# def is_lora_repo(repo: str) -> bool:
#     try:
#         hf_hub_download(repo, "adapter_config.json", token=HF_TOKEN); return True
#     except Exception:
#         return osp.exists(osp.join(repo, "adapter_config.json"))

# def extract_question(text: str) -> str:
#     m = re.search(r"Question:(.*?)Reasoning:", text, re.S)
#     return (m.group(1) if m else text).strip()

# def load_base_and_merge(adapter_repo: str, tokenizer):
#     cfg = json.load(open(hf_hub_download(adapter_repo, "adapter_config.json", token=HF_TOKEN)))
#     base = AutoModelForCausalLM.from_pretrained(cfg["base_model_name_or_path"],
#                                                 device_map="auto", token=HF_TOKEN)
#     base.resize_token_embeddings(len(tokenizer))
#     merged = PeftModel.from_pretrained(base, adapter_repo, token=HF_TOKEN,
#                                        is_trainable=True).merge_and_unload()
#     return merged

# # ── data ──────────────────────────────────────────────────────────────────────
# df = pd.read_csv(args.train_csv).iloc[:2000, :]
# prompts = [extract_question(t) for t in df["prefix"]]
# train_ds, eval_ds = Dataset.from_dict({"prompt": prompts}).train_test_split(0.1, seed=42).values()


# tok = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
# tok.pad_token, tok.padding_side = tok.eos_token, "left"

# # 修改点1: 使用add_tokens而不是add_special_tokens
# # 对于Qwen，这些token应该作为普通token添加
# tok.add_tokens(SPECIAL_TOKENS)  # 将special_tokens直接作为普通token添加

# print("Special tokens in tokenizer:", tok.special_tokens_map)

# for t in SPECIAL_TOKENS:
#     tid = tok.convert_tokens_to_ids(t)
#     print(f"{t} → id {tid}")

# for t in SPECIAL_TOKENS:
#     print(f"tokenize({t}) →", tok.tokenize(t))


# # ── model ─────────────────────────────────────────────────────────────────────
# if is_lora_repo(args.model_name):
#     model = load_base_and_merge(args.model_name, tok)
# else:
#     model = AutoModelForCausalLM.from_pretrained(args.model_name,
#                                                  device_map="auto", token=HF_TOKEN)
#     model.resize_token_embeddings(len(tok))

# if args.use_lora:
#     lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
#                           lora_dropout=args.lora_dropout, bias="none",
#                           task_type="CAUSAL_LM",
#                           target_modules=["q_proj","k_proj","v_proj","o_proj",
#                                           "gate_proj","up_proj","down_proj"])
#     model = get_peft_model(model, lora_cfg)

# # ── reward ────────────────────────────────────────────────────────────────────
# def reward_fn(prompts, completions, **kwargs):
#     return [1.0 if sum(tok in c for tok in SPECIAL_TOKENS) == 1 else 0.0 for c in completions]


# class QwenTokenDecoderWrapper:
#     """完全委托模式的包装器，只覆盖解码方法"""
    
#     def __init__(self, tokenizer):
#         # 保存原始tokenizer
#         self._tokenizer = tokenizer
        
#         # 获取tokenizer的类型
#         self._is_qwen = "Qwen" in tokenizer.__class__.__name__
        
#     def __getattr__(self, name):
#         """所有未明确定义的方法都直接委托给原始tokenizer"""
#         return getattr(self._tokenizer, name)
        
#     def __call__(self, *args, **kwargs):
#         """直接委托给原始tokenizer"""
#         return self._tokenizer(*args, **kwargs)
    
#     def batch_decode(self, *args, **kwargs):
#         """确保特殊token不被跳过"""
#         kwargs["skip_special_tokens"] = False
#         return self._tokenizer.batch_decode(*args, **kwargs)
    
#     def decode(self, *args, **kwargs):
#         """确保特殊token不被跳过"""
#         kwargs["skip_special_tokens"] = False
#         return self._tokenizer.decode(*args, **kwargs)


        
# # ── GRPO config & trainer ────────────────────────────────────────────────────
# cfg = GRPOConfig(
#     output_dir=args.output_dir,
#     num_iterations=args.num_iterations,
#     num_generations=args.num_generations,
#     per_device_train_batch_size=args.per_device_train_batch_size,
#     gradient_accumulation_steps=args.gradient_accumulation_steps,
#     learning_rate=args.learning_rate,
#     max_prompt_length=256,
#     max_completion_length=128,
#     temperature=0.8, top_k=50, top_p=0.92, repetition_penalty=1.1,
#     log_completions=True,
#     logging_strategy="steps", logging_steps=20,
#     lr_scheduler_type="cosine")

# # 修改点4: 使用优化后的QwenTokenDecoderWrapper
# trainer = GRPOTrainer(model=model, processing_class=QwenTokenDecoderWrapper(tok), args=cfg,
#                       train_dataset=train_ds, eval_dataset=eval_ds,
#                       reward_funcs=reward_fn)
# trainer.train()

# # ── save ──────────────────────────────────────────────────────────────────────
# model_path = osp.join(args.output_dir, "final_model")
# model.save_pretrained(model_path)
# tok.save_pretrained(model_path)
# json.dump(vars(args), open(osp.join(args.output_dir, "training_config.json"), "w"), indent=2)
# print(f"Done. Saved to {model_path}")