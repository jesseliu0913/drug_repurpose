#!/bin/bash

# for prompt_type in fcot phenotype cot gene fraw raw raw3; do
#   CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py \
#     --model_name "meta-llama/Llama-3.2-1B-Instruct" \
#     --output_path "results/llama32_1b/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 1 > "./log/llama32_1b_${prompt_type}.log" 2>&1 & 
# done

for prompt_type in fcot phenotype cot gene fraw raw raw3; do
  CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_path "uncertainty_results/llama32_1b/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 10 > "./log/llama32_1b_${prompt_type}.log" 2>&1 & 
done




