import os
import json
import random
import torch
import argparse
import numpy as np
import pandas as pd
import wandb
import logging
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

set_seed(42)

parser = argparse.ArgumentParser(description="Casual Tuning based on case report")
parser.add_argument("--model", type=str, default=None, help="Set model weights")
parser.add_argument("--task", type=str, default=None, help="Set Task Name")
parser.add_argument("--batch_size", type=int, default=8, help="Set Batch Size")
parser.add_argument("--training_data", type=str, default=None, help="Set Training Data")
parser.add_argument("--tunning_type", type=str, default=None, help="Set Tunning Type")

args = parser.parse_args()


user_token = os.getenv("HF_API_TOKEN")
if args.tunning_type == "baseline":
    if args.training_data == 'kpath':
        train_data = pd.read_csv("../grpo_part_path/k_path/train_grpo_baseline.csv")
    elif args.training_data == 'pagerank':
        train_data = pd.read_csv("../grpo_part_path/page_rank/train_grpo_baseline.csv")
    elif args.training_data == 'balance_path':
        train_data = pd.read_csv("../grpo_part_path/path_filter/train_grpo_baseline.csv")
elif args.tunning_type == "naive":
    if args.training_data == 'kpath':
        train_data = pd.read_csv("../grpo_part_path/k_path/train_grpo_naive.csv")
    elif args.training_data == 'pagerank':
        train_data = pd.read_csv("../grpo_part_path/page_rank/train_grpo_naive.csv")
    elif args.training_data == 'balance_path':
        train_data = pd.read_csv("../grpo_part_path/path_filter/train_grpo_naive.csv")
else:
    print("Tuning Type Wrong")



prompts = train_data['prefix'].tolist()
dataset = Dataset.from_dict({"text": prompts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=user_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, test_prompt, every_n_steps=20):
        self.tokenizer = tokenizer
        self.test_prompt = test_prompt
        self.every_n_steps = every_n_steps
        
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            input_ids = self.tokenizer.encode(self.test_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7
                )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            print(f"\nStep {state.global_step} generation test:")
            print(f"Prompt: {self.test_prompt}")
            print(f"Output: {generated_text}\n")

test_prompt = prompts[0][:100]  
print(f"Test prompt for generation: {test_prompt}")

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="np"
    )
    return result

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing datasets"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    token=user_token,
    device_map="auto",
    rope_scaling={"type": "dynamic", "factor": 32.0}
)
model.resize_token_embeddings(len(tokenizer))

for param in model.parameters():
    param.requires_grad = False

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

model.get_input_embeddings().weight.requires_grad = True
model.config.pad_token_id = tokenizer.pad_token_id
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

generation_callback = GenerationCallback(tokenizer, test_prompt)
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.005
)

training_args = TrainingArguments(
    output_dir=f"./model_weights/{args.task}-{args.training_data}-{args.tunning_type}",
    eval_strategy="steps",
    eval_steps=25,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=1,
    warmup_ratio=0.05,
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    optim="adamw_torch",
    report_to="wandb",
    max_grad_norm=1.0,
    save_total_limit=3, 
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    callbacks=[generation_callback, early_stopping_callback],
)

trainer.train()

model.save_pretrained(f"./model_weights/{args.task}-{args.training_data}-final{args.tunning_type}")
tokenizer.save_pretrained(f"./model_weights/{args.task}-{args.training_data}-final{args.tunning_type}")

# CUDA_VISIBLE_DEVICES=0 nohup python train.py > ./log/train_1b.log 2>&1 &