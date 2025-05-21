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
# Need to rerun
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "fcot" --shuffle_num 1 > "./log/llama32_1b_tune_fcot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_1b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_1b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_1b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "fraw" --shuffle_num 1 > "./log/llama32_1b_tune_fraw.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-ddbaseline" --output_path "results/llama32_1b_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_1b_tune_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-cold" --output_path "results/llama32_1b_pagerank/" --prompt_type "fraw" --shuffle_num 1 > "./log/llama32_1b_tune_fraw.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.1-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-cold" --output_path "results/llama32_1b_pagerank/" --prompt_type "fcot" --shuffle_num 1 > "./log/llama32_1b_tune_fcot.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-cold" --output_path "results/llama32_1b_pagerank/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_1b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-cold" --output_path "results/llama32_1b_pagerank/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_1b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-cold" --output_path "results/llama32_1b_pagerank/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_1b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-cold" --output_path "results/llama32_1b_pagerank/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_1b_tune_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-kpath" --output_path "results/llama32_1b_kpath/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_1b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-kpath" --output_path "results/llama32_1b_kpath/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_1b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-kpath" --output_path "results/llama32_1b_kpath/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_1b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-kpath" --output_path "results/llama32_1b_kpath/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_1b_tune_raw.log" 2>&1 &

# HongxuanLi/llama32-1b-kpath-grpo
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "HongxuanLi/llama32-1b-kpath-grpo" --output_path "results/llama32_1b_kpath_grpo/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_1b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "HongxuanLi/llama32-1b-kpath-grpo" --output_path "results/llama32_1b_kpath_grpo/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_1b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "HongxuanLi/llama32-1b-kpath-grpo" --output_path "results/llama32_1b_kpath_grpo/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_1b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "HongxuanLi/llama32-1b-kpath-grpo" --output_path "results/llama32_1b_kpath_grpo/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_1b_tune_raw.log" 2>&1 &

# JesseLiu/llama32-1b-pagerank-partial
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-pagerank-partial" --output_path "results/llama32_1b_pagerank_partial/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_1b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-pagerank-partial" --output_path "results/llama32_1b_pagerank_partial/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_1b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-pagerank-partial" --output_path "results/llama32_1b_pagerank_partial/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_1b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-1B-Instruct" --adapter_name "JesseLiu/llama32-1b-pagerank-partial" --output_path "results/llama32_1b_pagerank_partial/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_1b_tune_raw.log" 2>&1 &