#!/bin/bash
CUDA_VISIBLE_DEVICES=7 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_path "results/llama32_1b/" \
    --prompt_type "gene" \
    --shuffle_num 1 > ./log/llama32_1b.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results/llama32_3b/" \
    --prompt_type "gene" \
    --shuffle_num 1 > ./log/llama32_3b.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_path "results/llama32_1b/" \
    --prompt_type "fraw" \
    --shuffle_num 1 > ./log/llama32_1b_fraw.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results/llama32_3b/" \
    --prompt_type "fraw" \
    --shuffle_num 1 > ./log/llama32_3b_fraw.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_path "results/llama32_1b/" \
    --prompt_type "gene" \
    --shuffle_num 1 > ./log/llama32_1b.log 2>&1 

