#!/bin/bash

# for prompt_type in fcot phenotype cot gene fraw raw raw3; do
#   CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py \
#     --model_name "JesseLiu/llama32-1b-lora_cot" \
#     --output_path "results/llama32_3b_loracot/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 1 > "./log/llama32_3b_tune_${prompt_type}.log" 2>&1 &
# done

# for prompt_type in fcot phenotype cot gene fraw raw; do
#   CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py \
#     --model_name "meta-llama/Llama-3.2-3B-Instruct" \
#     --adapter_name "JesseLiu/llama32-3b-ddbaseline" \
#     --output_path "results/llama32_3b_baseline/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 1 > "./log/llama32_3b_tune_${prompt_type}.log" 2>&1 
# done


# for prompt_type in fcot phenotype cot gene fraw raw; do
#   CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py \
#     --model_name "meta-llama/Llama-3.2-3B-Instruct" \
#     --adapter_name "JesseLiu/llama32-3b-ddbaseline" \
#     --output_path "results/llama32_3b_baseline/" \
#     --prompt_type "$prompt_type" \
#     --shuffle_num 1 > "./log/llama32_3b_tune_${prompt_type}.log" 2>&1 
# done

CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-ddbaseline" --output_path "results/llama32_3b_baseline/" --prompt_type "fcot" --shuffle_num 1 > "./log/llama32_3b_tune_fcot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-ddbaseline" --output_path "results/llama32_3b_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-ddbaseline" --output_path "results/llama32_3b_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-ddbaseline" --output_path "results/llama32_3b_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-ddbaseline" --output_path "results/llama32_3b_baseline/" --prompt_type "fraw" --shuffle_num 1 > "./log/llama32_3b_tune_fraw.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-ddbaseline" --output_path "results/llama32_3b_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_cold/" --prompt_type "fraw" --shuffle_num 1 > "./log/llama32_3b_tune_fraw.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_cold/" --prompt_type "fcot" --shuffle_num 1 > "./log/llama32_3b_tune_fcot.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath" --output_path "results/llama32_3b_kpath/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath" --output_path "results/llama32_3b_kpath/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath" --output_path "results/llama32_3b_kpath/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath" --output_path "results/llama32_3b_kpath/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &

# HongxuanLi/llama32-3b-kpath-grpo
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "HongxuanLi/llama32-3b-kpath-grpo" --output_path "results/llama32_3b_kpath_grpo/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "HongxuanLi/llama32-3b-kpath-grpo" --output_path "results/llama32_3b_kpath_grpo/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "HongxuanLi/llama32-3b-kpath-grpo" --output_path "results/llama32_3b_kpath_grpo/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "HongxuanLi/llama32-3b-kpath-grpo" --output_path "results/llama32_3b_kpath_grpo/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &
