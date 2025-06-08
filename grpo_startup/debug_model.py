import os
from transformers import AutoConfig, AutoModelForCausalLM

BASE_NAME = "JesseLiu/llama32-1b-balancepath-partial-baseline"
TOKEN     = os.getenv("HF_TOKEN")

# 1) Load the config alone, without instantiating the model
cfg = AutoConfig.from_pretrained(BASE_NAME, use_auth_token=TOKEN)

# 2) Print out any fields that might be None
candidates = [
    "model_parallel_backend",
    "parallelize",
    "tensor_parallel_degree",
    "tensor_parallel_group_size",
    "tensor_parallel_split_count",
    "tensor_parallel_split_rank",
]
print(">>> Checking parallel‐related config fields:")
for key in candidates:
    val = getattr(cfg, key, "(attribute missing)")
    print(f"  {key!r} = {val!r}")
