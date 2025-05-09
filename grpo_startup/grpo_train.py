import os
import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import sys
import argparse
import torch
from peft import PeftModel, LoraConfig, get_peft_model
import json
from huggingface_hub import hf_hub_download
import os.path as osp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parse command line arguments
parser = argparse.ArgumentParser(description='GRPO fine-tuning for language models')
parser.add_argument('--model_name', type=str, default="JesseLiu/llama32-3b-cold", 
                    help='Model name or path')
parser.add_argument('--output_dir', type=str, default="llama3.2-grpo-out", 
                    help='Directory to save the model')
parser.add_argument('--per_device_train_batch_size', type=int, default=2, 
                    help='Batch size per device for training')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                    help='Number of gradient accumulation steps')
parser.add_argument('--num_iterations', type=int, default=5, 
                    help='Number of GRPO iterations')
parser.add_argument('--learning_rate', type=float, default=1e-5, 
                    help='Learning rate')
parser.add_argument('--num_generations', type=int, default=4, 
                    help='Number of generations per prompt')
parser.add_argument('--use_lora', action='store_true', 
                    help='Use LoRA for fine-tuning')
parser.add_argument('--lora_r', type=int, default=16, 
                    help='LoRA r dimension')
parser.add_argument('--lora_alpha', type=int, default=32, 
                    help='LoRA alpha parameter')
parser.add_argument('--lora_dropout', type=float, default=0.05, 
                    help='LoRA dropout probability')
args = parser.parse_args()

user_token = os.environ.get("HF_API_TOKEN")
train_data = pd.read_csv("../grpo_path/train_grpo.csv")

# Function to check if model is already a LoRA model
def is_lora_model(model_name):
    try:
        # Try to download adapter_config.json from the model repo
        try:
            config_path = hf_hub_download(model_name, "adapter_config.json", token=user_token)
            return True
        except:
            # Check if the model has a local adapter_config.json
            if osp.exists(osp.join(model_name, "adapter_config.json")):
                return True
            return False
    except:
        return False

# Extract just the question part from each prefix
def extract_question(prefix):
    match = re.search(r'Question:(.*?)Reasoning:', prefix, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prefix

# Apply the extraction function to each prefix
prompts = [extract_question(prefix) for prefix in train_data["prefix"].tolist()]
print("Using questions-only for training")

dataset = Dataset.from_dict({"prompt": prompts})
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, token=user_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

new_special_tokens = ['<degd>', '<ddd>', '<decgd>', '<demgd>', '<debgd>', '<dppd>', '<dpd>']
tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

# Check if the model is already a LoRA model
is_lora = is_lora_model(model_name)
print(f"Detected model type: {'LoRA adapter' if is_lora else 'Original model'}")

if is_lora:
    # Load the LoRA model
    print("Loading LoRA model for continued training...")
    base_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=user_token),
        model_name,
        token=user_token
    )
    
    # For continued LoRA training, we'll merge the LoRA weights and continue training
    print("Merging LoRA weights with base model for continued training...")
    model = base_model.merge_and_unload()
    
    # If use_lora flag is set, we'll prepare a new LoRA configuration for further training
    if args.use_lora:
        print("Setting up new LoRA adapter for continued training...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        print("New LoRA adapter configured successfully")
else:
    # Load original model
    print("Loading original model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=user_token)
    
    # Apply LoRA if requested
    if args.use_lora:
        print("Setting up LoRA fine-tuning for original model...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        print("LoRA adapter configured successfully")

model.resize_token_embeddings(len(tokenizer))

def reward_one_type_only(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        types_present = {tok for tok in new_special_tokens if tok in comp}
        rewards.append(1.0 if len(types_present) == 1 else 0.0)
    return rewards

# Update GRPOConfig with supported parameters only
training_args = GRPOConfig(
    output_dir=args.output_dir,
    num_iterations=args.num_iterations,
    num_generations=args.num_generations,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    max_prompt_length=256,  # Reduced from 512 based on data
    max_completion_length=128,  # Increased from 64 for full reasoning paths
    
    # Generation parameters
    temperature=0.8,  # Increased from 0.7 for more exploration
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.1,  # Added to prevent repetitive text
    
    # Training settings
    logging_strategy="steps",
    logging_steps=20,
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    reward_funcs=reward_one_type_only,
)

trainer.train()

# Save the final model
model_path = os.path.join(args.output_dir, "final_model")
if args.use_lora or is_lora:
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(model_path)
    else:
        # For merged models
        model.save_pretrained(model_path)
else:
    model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save training configuration
with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
    json.dump({
        "model_name": model_name,
        "is_lora_model": is_lora,
        "used_lora_finetuning": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
        "lora_dropout": args.lora_dropout if args.use_lora else None,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "num_generations": args.num_generations,
    }, f, indent=2)
