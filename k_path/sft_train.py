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
path_data = load_dataset(f"Tassy24/K-Paths-inductive-reasoning-{args.data_type}")

train_data = path_data["train"]
prompts = []
if args.data_type == "ddinter":
    for i in range(len(train_data)):
        reasoning_path = train_data[i]['path_str'].split("\n")[0:6]
        prefix = (
            f"Question: What is the interaction severity between {train_data[i]['drug1_name']} and {train_data[i]['drug2_name']}?\n\n",
            f"Choices:[Major, Moderate, Minor, No Interaction]\n\n",
            f"Possible Reasoning Chains:\n",
            # f"Drug ({train_data[i]['drug1_name']}): {train_data[i]['drug1_desc']}\n",
            # f"Drug ({train_data[i]['drug2_name']}): {train_data[i]['drug2_desc']}\n",
            # f"From the knowledge graph:\n",
            f"{['\n'.join(reasoning_path)]}\n\n",
            f"Answer: {train_data[i]['label']}"
        )
        prompts.append("".join(prefix))
elif args.data_type == "drugbank":
    for i in range(len(train_data)):
        reasoning_path = train_data[i]['path_str'].split("\n")[0:6]
        prefix = (
            f"Question: What is the pharmacological interaction between {train_data[i]['drug1_name']} and {train_data[i]['drug2_name']}?\n\n",
            f"Possible Reasoning Chains:\n",
            # f"Drug ({train_data[i]['drug1_name']}): {train_data[i]['drug1_desc']}\n",
            # f"Drug ({train_data[i]['drug2_name']}): {train_data[i]['drug2_desc']}\n",
            f"{['\n'.join(reasoning_path)]}\n\n",
            f"Answer: {train_data[i]['label']}"
        )
        prompts.append("".join(prefix))
elif args.data_type == "pharmaDB":
    for i in range(len(train_data)):
        reasoning_path = train_data[i]['path_str'].split("\n")[0:6]
        prefix = (
            f"Question: What is the therapeutic relationship between {train_data[i]['drug_name']} and {train_data[i]['disease_name']}?\n\n",
            f"Choices:[disease-modifying, palliates, non-indication]\n\n",
            f"Possible Reasoning Chains:\n",
            # f"Drug ({train_data[i]['drug_name']}): {train_data[i]['drug_desc']}\n",
            # f"Disease ({train_data[i]['disease_name']}): {train_data[i]['disease_desc']}\n",
            # f"From the knowledge graph:\n",
            f"{['\n'.join(reasoning_path)]}\n\n",
            f"Answer: {train_data[i]['label']}"
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
    learning_rate=2e-4,
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

# CUDA_VISIBLE_DEVICES=0 nohup python train.py > ./log/train_1b.log 2>&1 &
"""
Question: What is the interaction severity between Lansoprazole and Lapatinib?

Reasoning:
Drug (Lansoprazole): Lansoprazole marketed under the brand Prevacid, is a proton pump inhibitor (PPI) and is structurally classified as a substituted benzimidazole. It reduces gastric acid secretion by targeting gastric H,K-ATPase pumps and is thus effective at promoting healing in ulcerative diseases, and treating gastroesophageal reflux disease (GERD) along with other pathologies caused by excessive acid secretion.
Drug (Lapatinib): Lapatinib is an anti-cancer drug developed by GlaxoSmithKline (GSK) as a treatment for solid tumours such as breast and lung cancer. It was approved by the FDA on March 13, 2007, for use in patients with advanced metastatic breast cancer in conjunction with the chemotherapy drug capecitabine. Lapatinib is a human epidermal growth factor receptor type 2 (HER2/ERBB2) and epidermal growth factor receptor (HER1/EGFR/ERBB1) tyrosine kinases inhibitor. It binds to the intracellular phosphorylation domain to prevent receptor autophosphorylation upon ligand binding.
From the knowledge graph:
Lansoprazole (Compound) binds ABCB1 (Gene) and ABCB1 (Gene) is bound by Lapatinib (Compound)
Lansoprazole (Compound) downregulates SUPV3L1 (Gene) and SUPV3L1 (Gene) is downregulated by Lapatinib (Compound)
Lansoprazole (Compound) causes Nail disorder (Side Effect) and Nail disorder (Side Effect) is caused by Lapatinib (Compound)
Lansoprazole may lead to a major life threatening interaction when taken with Neratinib and Neratinib may cause a moderate interaction that could exacerbate diseases when taken with Lapatinib
Lansoprazole may cause a moderate interaction that could exacerbate diseases when taken with Aprepitant and Aprepitant may cause a moderate interaction that could exacerbate diseases when taken with Lapatinib
Lansoprazole may cause a minor interaction that can limit clinical effects when taken with Axitinib and Axitinib may cause a moderate interaction that could exacerbate diseases when taken with Lapatinib
Lansoprazole may cause a moderate interaction that could exacerbate diseases when taken with Fluvoxamine and Fluvoxamine may cause a moderate interaction that could exacerbate diseases when taken with Lapatinib
Lansoprazole may cause a moderate interaction that could exacerbate diseases when taken with Fosaprepitant and Fosaprepitant may cause a moderate interaction that could exacerbate diseases when taken with Lapatinib
Lansoprazole may lead to a major life threatening interaction when taken with Erlotinib and Erlotinib may cause a moderate interaction that could exacerbate diseases when taken with Lapatinib

Answer: Moderate
"""