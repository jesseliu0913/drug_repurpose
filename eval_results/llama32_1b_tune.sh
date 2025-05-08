#!/bin/bash

# for prompt_type in fcot phenotype cot gene fraw raw raw3; do
#   CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py \
#     --model_name "JesseLiu/llama32-1b-lora_cot" \
#     --output_path "results/llama32_1b_loracot/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 1 > "./log/llama32_1b_${prompt_type}_tune.log" 2>&1 &
# done

# for prompt_type in fcot phenotype cot gene fraw raw raw3; do
#   CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py \
#     --model_name "JesseLiu/llama32-1b-lora_cot" \
#     --output_path "uncertainty_results/llama32_1b_loracot/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 10 > "./log/llama32_1b_${prompt_type}_uc.log" 2>&1 &
# done

CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-3B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "fcot" --shuffle_num 1 > "./log/llama32_1b_tune_fcot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-3B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_1b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-3B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_1b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-3B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_1b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-3B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "fraw" --shuffle_num 1 > "./log/llama32_1b_tune_fraw.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-3B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_1b_tune_raw.log" 2>&1 &