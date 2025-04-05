#!/bin/bash

for prompt_type in raw; do
    echo $prompt_type
    nohup python eval_gemini.py \
        --output_path "results/gemini/" \
        --prompt_type "$prompt_type" \
        --shuffle_num 1 > "./log/gemini_${prompt_type}.log" 2>&1 
done