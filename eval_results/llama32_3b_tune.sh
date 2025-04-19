#!/bin/bash

# for prompt_type in fcot phenotype cot gene fraw raw raw3; do
#   CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py \
#     --model_name "JesseLiu/llama32-1b-lora_cot" \
#     --output_path "results/llama32_3b_loracot/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 1 > "./log/llama32_3b_tune_${prompt_type}.log" 2>&1 &
# done

for prompt_type in fcot phenotype cot gene fraw raw raw3; do
  CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py \
    --model_name "JesseLiu/llama32-1b-lora_cot" \
    --output_path "uncertainty_results/llama32_3b_loracot/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 10 > "./log/llama32_3b_tune_${prompt_type}_uc.log" 2>&1 &
done