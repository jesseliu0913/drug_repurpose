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

model_name = "meta-llama/Llama-2-7b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    device_map="auto"
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  
)

training_args = TrainingArguments(
    output_dir="./llama32-1b-model",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    learning_rate=1e-5, 
    weight_decay=0.01,
    fp16=True,
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    warmup_steps=100,
    load_best_model_at_end=True,
    gradient_accumulation_steps=16,  
    report_to="wandab",
    gradient_checkpointing=True,  
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
trainer.save_model("./llama32-1b-model")