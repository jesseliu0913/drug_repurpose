#!/bin/bash


CUDA_VISIBLE_DEVICES=4 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_path "results/llama32_1b/" \
    --shuffle_num 10 > ./log/llama32_1b.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results/llama32_3b/" \
    --shuffle_num 10 > ./log/llama32_3b.log 2>&1 &

