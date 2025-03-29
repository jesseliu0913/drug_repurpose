#!/bin/bash

mkdir -p ./log

nohup python gemini_eval.py \
    --output_path "results/gemini/" \
    --prompt_type "cot" \
    --shuffle_num 1 > ./log/gemini.log 2>&1

nohup python gemini_eval.py \
    --output_path "results/gemini/" \
    --prompt_type "fcot" \
    --shuffle_num 1 > ./log/gemini.log 2>&1

nohup python gemini_eval.py \
    --output_path "results/gemini/" \
    --prompt_type "phenotype" \
    --shuffle_num 1 > ./log/gemini.log 2>&1

nohup python gemini_eval.py \
    --output_path "results/gemini/" \
    --prompt_type "gene" \
    --shuffle_num 1 > ./log/gemini.log 2>&1
