import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

set_seed(42)

ddgene_path = Path("data/ddgene.jsonl")
ddphenotype_path = Path("data/ddphenotype.jsonl")

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

ddgene_data = read_file(ddgene_path)
ddphenotype_data = read_file(ddphenotype_path)
combined_data = ddgene_data + ddphenotype_data
random.shuffle(combined_data)

prompts = [item['prompt'] for item in combined_data]
dataset = Dataset.from_dict({"text": prompts})
dataset = dataset.train_test_split(test_size=0.1)

model_name = "meta-llama/Llama-3.2-3B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_function(examples):
    texts = [text + tokenizer.eos_token for text in examples["text"]]
    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing datasets"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=token
)

# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
# )

for param in model.parameters():
    param.requires_grad = True

# model = get_peft_model(model, lora_config)

# for name, param in model.named_parameters():
#     if "lora" in name:  
#         if not param.requires_grad:
#             param.requires_grad = True

# model.print_trainable_parameters()  

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  
)

training_args = TrainingArguments(
    output_dir="./llama32-3b-model",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    learning_rate=1e-5, 
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    warmup_steps=100,
    load_best_model_at_end=True,
    gradient_accumulation_steps=16,  
    gradient_checkpointing=False,  
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./llama32-3b-model")

# CUDA_VISIBLE_DEIVCES=0,1,6,7 torchrun --nproc_per_node=2 instruct_tuning.py > llama32-3b.log 2>&1 &
