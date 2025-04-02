#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw raw; do
  CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results/llama32_3b/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 1 > "./log/llama32_3b_${prompt_type}.log" 2>&1 
done