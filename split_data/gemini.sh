#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw raw raw3; do
    nohup python -u eval_gemini.py \
        --output_path "results/gemini/" \
        --prompt_type "$prompt_type" \
        --shuffle_num 1 > "./log/gemini_${prompt_type}.log" 2>&1 
done
