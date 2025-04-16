#!/bin/bash


for prompt_type in fcot phenotype cot gene fraw raw raw3; do
  CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results/llama32_3b/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 1 > "./log/llama32_3b_${prompt_type}.log" 2>&1 &
done

# for prompt_type in fcot phenotype cot gene fraw raw raw3; do
#   CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py \
#     --model_name "meta-llama/Llama-3.2-3B-Instruct" \
#     --output_path "uncertainty_results/llama32_3b/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 10 > "./log/llama32_3b_${prompt_type}_uc.log" 2>&1 &
# done