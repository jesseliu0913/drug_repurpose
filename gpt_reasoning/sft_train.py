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
from datasets import load_dataset

set_seed(42)

parser = argparse.ArgumentParser(description="Casual Tuning based on case report")
parser.add_argument("--model", type=str, default=None, help="Set model weights")
parser.add_argument("--task", type=str, default=None, help="Set Task Name")
parser.add_argument("--data_type", type=str, default=None, help="Set Data Type")
parser.add_argument("--batch_size", type=int, default=1, help="Set Batch Size")
args = parser.parse_args()


user_token = os.getenv("HF_API_TOKEN")

assert args.data_type in ["ddinter", "drugbank", "pharmaDB"], "Invalid data type. Choose from ['ddinter', 'drugbank', 'pharmaDB']"
data_path = Path(f"./reasoning_data/{args.data_type}.csv")

train_data = pd.read_csv(data_path)
prompts = []
if args.data_type == "ddinter":
    for index, row in train_data.iterrows():
        label_to_letter = {
            "Major": "A",
            "Moderate": "B",
            "Minor": "C",
            "No Interaction": "D"
        }
        label = row['label']
        letter = label_to_letter.get(label, "")

        reasoning_path = row['reason_path']
        prefix = (
            f"Question: What is the interaction severity between {row['drug1_name']} and {row['drug2_name']}?\n\n",
            f"Choices:[Major, Moderate, Minor, No Interaction]\n\n",
            f"Reasoning:\n",
            f"{reasoning_path}\n\n",
            f"Answer: {letter}.{label}"
        )
        prompts.append("".join(prefix))
elif args.data_type == "drugbank":
    for index, row in train_data.iterrows():
        reasoning_path = row['reason_path']
        prefix = (
            f"Question: What is the pharmacological interaction between {row['drug1_name']} and {row['drug2_name']}?\n\n",
            f"Reasoning:\n",
            f"{reasoning_path}\n\n",
            f"Answer: {row['label']}"
        )
        prompts.append("".join(prefix))
elif args.data_type == "pharmaDB":
    for index, row in train_data.iterrows():
        label_to_letter = {
            "Disease-modifying": "A",
            "Palliates": "B",
            "Non-indication": "C",
        }
        label = row['label']
        letter = label_to_letter.get(label, "")
        
        reasoning_path = row['reason_path']
        prefix = (
            f"Question: What is the therapeutic relationship between {row['drug_name']} and {row['disease_name']}?\n\n",
            f"Choices:[Disease-modifying, Palliates, Non-indication]\n\n",
            f"Reasoning:\n",
            f"{reasoning_path}\n\n",
            f"Answer: {letter}.{label}"
        )
        prompts.append("".join(prefix))

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

test_prompt = prompts[0]
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
if args.model == "Qwen/Qwen2.5-3B":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=user_token,
        # device_map="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=user_token,
        # device_map="auto",
        rope_scaling={"type": "dynamic", "factor": 32.0}
    )

for param in model.parameters():
    param.requires_grad = False

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

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
    # output_dir=f"./model_weights/{args.task}-{args.data_type}",
    eval_strategy="steps",
    eval_steps=25,
    logging_dir="./logs",
    logging_steps=5,
    # save_strategy="none",
    # save_steps=200,
    learning_rate=4e-5,
    weight_decay=0.01,
    fp16=True,
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
    dataloader_num_workers=6,
    # deepspeed="ds_config_zero3.json",
    ddp_find_unused_parameters=False,
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

model.save_pretrained(f"./model_weights/{args.task}-final{args.data_type}")
tokenizer.save_pretrained(f"./model_weights/{args.task}-final{args.data_type}")

# CUDA_VISIBLE_DEVICES=0 nohup python train.py > ./logs/train_1b.log 2>&1 &
