#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw; do
  CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py \
    --model_name "JesseLiu/llama32-3b_raw_cot" \
    --output_path "results/llama32_3b_rawcot/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 1 > "./log/llama32_3b_${prompt_type}.log" 2>&1 
done

CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py \
  --model_name "JesseLiu/llama32-3b-lora_cot" \
  --output_path "results/llama32_3b_loracot/" \
  --prompt_type "raw" \
  --shuffle_num 1 > "./log/llama32_3b_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py \
  --model_name "JesseLiu/llama32-3b-lora_cot" \
  --output_path "results/llama32_3b_loracot/" \
  --prompt_type "gene" \
  --shuffle_num 1 > "./log/llama32_3b_gene.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py \
  --model_name "JesseLiu/llama32-3b-lora_cot" \
  --output_path "results/llama32_3b_loracot/" \
  --prompt_type "cot" \
  --shuffle_num 1 > "./log/llama32_3b_cot.log" 2>&1 &