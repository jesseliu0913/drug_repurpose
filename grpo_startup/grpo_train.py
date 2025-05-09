import os
import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parse command line arguments
parser = argparse.ArgumentParser(description='GRPO fine-tuning for Llama 3.2 models')
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
args = parser.parse_args()

user_token = os.environ.get("HF_API_TOKEN")
train_data = pd.read_csv("../grpo_path/train_grpo.csv")

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

model = AutoModelForCausalLM.from_pretrained(model_name, token=user_token)
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
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    reward_funcs=reward_one_type_only,
)

trainer.train()

# Save the final model
model_path = os.path.join(args.output_dir, "final_model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
