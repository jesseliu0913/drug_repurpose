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

user_token = os.getenv("HF_API_TOKEN")
train_data = Path("data/train.jsonl")

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = read_file(train_data)
random.shuffle(train_data)
combined_data = train_data

prompts = [item['prompt'] for item in combined_data]
dataset = Dataset.from_dict({"text": prompts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model_name = "meta-llama/Llama-3.2-3B-Instruct"  
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
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
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
for param in model.parameters():
    param.requires_grad = False

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

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
    output_dir="./llama32-3b-lora-model",
    evaluation_strategy="steps",
    eval_steps=25,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="steps",
    save_steps=25,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
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

model.save_pretrained("./llama32-3b-lora-adapter-final")
print("LoRA fine-tuning completed!")

# CUDA_VISIBLE_DEVICES=1 nohup python lora_tuning.py > ./log/lora_tunning_3b.log 2>&1 &