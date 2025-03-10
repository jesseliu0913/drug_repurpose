#!/bin/bash


CUDA_VISIBLE_DEVICES=5 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --dataset "mandarjoshi/trivia_qa" \
    --subset "rc" \
    --shuffle_num 10 \
    --output_path "results/llama32_1b/trivia_qa/" > ./log/llama32_1b_trivia.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --dataset "rajpurkar/squad" \
    --subset "None" \
    --shuffle_num 10 \
    --output_path "results/llama32_3b/squad_qa/" > ./log/llama32_3b_squad.log 2>&1 &

