#!/bin/bash

for prompt_type in fcot phenotype cot gene fraw raw raw3; do
  CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "$prompt_type" \
    --shuffle_num 1 > "./log/qwq_32b_${prompt_type}.log" 2>&1 
done

CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "cot" \
    --shuffle_num 1 > "./log/qwq_32b_cot.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "gene" \
    --shuffle_num 1 > "./log/qwq_32b_gene.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "raw3" \
    --shuffle_num 1 > "./log/qwq_32b_raw3.log" 2>&1 &



CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "raw" \
    --shuffle_num 1 > "./log/qwq_32b_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "phenotype" \
    --shuffle_num 1 > "./log/qwq_32b_phenotype.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py \
    --model_name "Qwen/QwQ-32B" \
    --output_path "results/qwq_32b/" \
    --prompt_type "fcot" \
    --shuffle_num 1 > "./log/qwq_32b_fcot.log" 2>&1 &

# full qwq
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath" --output_path "results/qwq7b_kpath/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_kpath_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath" --output_path "results/qwq7b_kpath/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_kpath_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath" --output_path "results/qwq7b_kpath/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_kpath_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath" --output_path "results/qwq7b_kpath/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_kpath_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank" --output_path "results/qwq7b_pagerank/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_pagerank_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank" --output_path "results/qwq7b_pagerank/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_pagerank_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank" --output_path "results/qwq7b_pagerank/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_pagerank_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank" --output_path "results/qwq7b_pagerank/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_pagerank_raw.log" 2>&1 &

# part qwq
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial" --output_path "results/qwq7b_kpath_partial/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_kpath_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial" --output_path "results/qwq7b_kpath_partial/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_kpath_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial" --output_path "results/qwq7b_kpath_partial/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_kpath_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial" --output_path "results/qwq7b_kpath_partial/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_kpath_raw.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial" --output_path "results/qwq7b_pagerank_partial/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_pagerank_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial" --output_path "results/qwq7b_pagerank_partial/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_pagerank_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial" --output_path "results/qwq7b_pagerank_partial/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_pagerank_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial" --output_path "results/qwq7b_pagerank_partial/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_pagerank_raw.log" 2>&1 &


CUDA_VISIBLE_DEVICES=3 python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial" --output_path "results/qwq7b_pagerank_partial/" --prompt_type "raw" --shuffle_num 1


# JesseLiu/qwen25-7b-pagerank-partial-naive
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-naive" --output_path "results/qwq25_7b_pagerank_partial_naive/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_pagerank_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-naive" --output_path "results/qwq25_7b_pagerank_partial_naive/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_pagerank_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-naive" --output_path "results/qwq25_7b_pagerank_partial_naive/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_pagerank_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-naive" --output_path "results/qwq25_7b_pagerank_partial_naive/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_pagerank_raw.log" 2>&1 &

# JesseLiu/qwen25-7b-kpath-partial-baseline
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial-baseline" --output_path "results/qwq25_7b_kpath_partial_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_kp_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial-baseline" --output_path "results/qwq25_7b_kpath_partial_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_kp_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial-baseline" --output_path "results/qwq25_7b_kpath_partial_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_kp_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-kpath-partial-baseline" --output_path "results/qwq25_7b_kpath_partial_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_kp_raw.log" 2>&1 &

# JesseLiu/qwen25-7b-pagerank-partial-baseline
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-baseline" --output_path "results/qwq25_7b_pagerank_partial_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/qwq_pg_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-baseline" --output_path "results/qwq25_7b_pagerank_partial_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/qwq_pg_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-baseline" --output_path "results/qwq25_7b_pagerank_partial_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/qwq_pg_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_model.py --model_name "Qwen/Qwen2.5-7B-Instruct" --adapter_name "JesseLiu/qwen25-7b-pagerank-partial-baseline" --output_path "results/qwq25_7b_pagerank_partial_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/qwq_pg_raw.log" 2>&1 &
