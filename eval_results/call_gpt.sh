#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw raw3; do
    nohup python -u eval_gpt.py \
        --output_path "uncertainty_results/gpt/" \
        --prompt_type "$prompt_type" \
        --shuffle_num 1 > "./log/gpt_${prompt_type}.log" 2>&1 &
done
nohup python -u eval_gpt.py \
    --output_path "uncertainty_results/gpt/" \
    --prompt_type "fcot" \
    --shuffle_num 1 > "./log/gpt_fcot.log" 2>&1 &

