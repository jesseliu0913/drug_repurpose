import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env_loader import load_env_vars

# Load environment variables
load_env_vars()

user_token = os.getenv("HF_API_TOKEN")

train_data = pd.read_csv("../grpo_path/train_grpo.csv")
prompts = train_data["prefix"].tolist()
dataset = Dataset.from_dict({"prompt": prompts})
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

model_name = "JesseLiu/llama32-3b-cold"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=user_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

new_special_tokens = ['<degd>', '<ddd>', '<decgd>', '<demgd>', '<debgd>', '<dppd>', '<dpd>']
tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

model = AutoModelForCausalLM.from_pretrained(model_name, token=user_token)
model.resize_token_embeddings(len(tokenizer))

def reward_one_type_only(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        types_present = {tok for tok in new_special_tokens if tok in comp}
        rewards.append(1.0 if len(types_present) == 1 else 0.0)
    return rewards

training_args = GRPOConfig(
    output_dir="llama3.2-grpo-out",
    num_iterations=5,
    num_generations=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Added gradient accumulation
    learning_rate=1e-5,
    max_prompt_length=512,
    max_completion_length=64,
    
    # Generation parameters
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_reward",
    greater_is_better=True,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    reward_funcs=reward_one_type_only,
)

trainer.train()

# Save the final model
model_path = os.path.join(training_args.output_dir, "final_model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
