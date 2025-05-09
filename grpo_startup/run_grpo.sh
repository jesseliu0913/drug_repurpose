#!/bin/bash

# Set the environment variable for HuggingFace token
export HF_API_TOKEN="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"  # Replace with your actual token or set before running

# Create log directory if it doesn't exist
mkdir -p /playpen/hongxuan/drug_repurpose/grpo_startup/logs

# === Push 1B model to HuggingFace Hub ===
echo "Pushing 1B model checkpoint to HuggingFace Hub..."
python /playpen/hongxuan/drug_repurpose/grpo_startup/push_model.py \
  --repo_name "JesseLiu/llama32-1b-cold" \
  --model_path "/playpen/hongxuan/drug_repurpose/grpo_startup/model_weights/llama32-1b-baseline-model/checkpoint-450" \
  > /playpen/hongxuan/drug_repurpose/grpo_startup/logs/push_1b.log 2>&1

# === GRPO training for 1B model ===
echo "Starting GRPO training for 1B model..."
CUDA_VISIBLE_DEVICES=0,1 python /playpen/hongxuan/drug_repurpose/grpo_startup/grpo_train.py \
  --model_name "JesseLiu/llama32-1b-cold" \
  --output_dir "/playpen/hongxuan/drug_repurpose/grpo_startup/model_weights/llama32-1b-grpo" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_iterations 5 \
  > /playpen/hongxuan/drug_repurpose/grpo_startup/logs/grpo_1b.log 2>&1

# === Push 3B model to HuggingFace Hub ===
echo "Pushing 3B model checkpoint to HuggingFace Hub..."
python /playpen/hongxuan/drug_repurpose/grpo_startup/push_model.py \
  --repo_name "JesseLiu/llama32-3b-cold" \
  --model_path "/playpen/hongxuan/drug_repurpose/grpo_startup/model_weights/llama32-3b-baseline-model/checkpoint-450" \
  > /playpen/hongxuan/drug_repurpose/grpo_startup/logs/push_3b.log 2>&1

# === GRPO training for 3B model ===
echo "Starting GRPO training for 3B model..."
CUDA_VISIBLE_DEVICES=2,3 python /playpen/hongxuan/drug_repurpose/grpo_startup/grpo_train.py \
  --model_name "JesseLiu/llama32-3b-cold" \
  --output_dir "/playpen/hongxuan/drug_repurpose/grpo_startup/model_weights/llama32-3b-grpo" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_iterations 5 \
  > /playpen/hongxuan/drug_repurpose/grpo_startup/logs/grpo_3b.log 2>&1

echo "GRPO training completed!"
