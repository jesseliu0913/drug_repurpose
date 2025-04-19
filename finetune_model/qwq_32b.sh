#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw raw raw3; do
  CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 1 > "./log/qwq_32b_${prompt_type}.log" 2>&1 
done

CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "cot" \
    --shuffle_num 1 > "./log/qwq_32b_cot.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "gene" \
    --shuffle_num 1 > "./log/qwq_32b_gene.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "raw3" \
    --shuffle_num 1 > "./log/qwq_32b_raw3.log" 2>&1 &


