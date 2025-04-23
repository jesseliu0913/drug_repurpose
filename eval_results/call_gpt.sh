#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw raw3; do
    nohup python -u eval_gpt.py \
        --output_path "uncertainty_results/gpt/" \
        --prompt_type "$prompt_type" \
        --shuffle_num 5 > "./log/gpt_${prompt_type}.log" 2>&1 &
done

