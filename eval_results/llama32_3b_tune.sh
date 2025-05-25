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
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-cold" --output_path "results/llama32_3b_pagerank/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 

# JesseLiu/llama32-3b-pagerank-partial
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial" --output_path "results/llama32_3b_pagerank_partial/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial" --output_path "results/llama32_3b_pagerank_partial/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial" --output_path "results/llama32_3b_pagerank_partial/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial" --output_path "results/llama32_3b_pagerank_partial/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &

# JesseLiu/llama32-3b-kpath-baseline
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &

# JesseLiu/llama32-3b-pagerank-partial-fullname
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw3.log" 2>&1 &

# JesseLiu/llama32-3b-kpath-partial-fullname
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw2.log" 2>&1 &


# JesseLiu/llama32-3b-pagerank-partial-abbr
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-abbr" --output_path "results/llama32_3b_pagerank_partial_abbr/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-abbr" --output_path "results/llama32_3b_pagerank_partial_abbr/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-abbr" --output_path "results/llama32_3b_pagerank_partial_abbr/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-abbr" --output_path "results/llama32_3b_pagerank_partial_abbr/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw3.log" 2>&1 &


# JesseLiu/llama32-3b-kpath-partial-abbr
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-abbr" --output_path "results/llama32_3b_kpath_partial_abbr/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-abbr" --output_path "results/llama32_3b_kpath_partial_abbr/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-abbr" --output_path "results/llama32_3b_kpath_partial_abbr/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-abbr" --output_path "results/llama32_3b_kpath_partial_abbr/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw2.log" 2>&1 &




# JesseLiu/llama32-3b-kpath-baseline
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-baseline" --output_path "results/llama32_3b_kpath_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw.log" 2>&1 &

# JesseLiu/llama32-3b-pagerank-baseline
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-baseline" --output_path "results/llama32_3b_pagerank_baseline/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-baseline" --output_path "results/llama32_3b_pagerank_baseline/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-baseline" --output_path "results/llama32_3b_pagerank_baseline/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-baseline" --output_path "results/llama32_3b_pagerank_baseline/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw1.log" 2>&1 &

# JesseLiu/llama32-3b-pagerank-naive
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-naive" --output_path "results/llama32_3b_pagerank_naive/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-naive" --output_path "results/llama32_3b_pagerank_naive/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-naive" --output_path "results/llama32_3b_pagerank_naive/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-naive" --output_path "results/llama32_3b_pagerank_naive/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw2.log" 2>&1 &

# JesseLiu/llama32-3b-kpath-naive
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-naive" --output_path "results/llama32_3b_kpath_naive/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-naive" --output_path "results/llama32_3b_kpath_naive/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-naive" --output_path "results/llama32_3b_kpath_naive/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-naive" --output_path "results/llama32_3b_kpath_naive/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw3.log" 2>&1 &

# JesseLiu/llama32-3b-pagerank-partial-fullname
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-pagerank-partial-fullname" --output_path "results/llama32_3b_pagerank_partial_fullname/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw3.log" 2>&1 &

# JesseLiu/llama32-3b-kpath-partial-fullname
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "phenotype" --shuffle_num 1 > "./log/llama32_3b_tune_phenotype2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "cot" --shuffle_num 1 > "./log/llama32_3b_tune_cot2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "gene" --shuffle_num 1 > "./log/llama32_3b_tune_gene2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_model.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --adapter_name "JesseLiu/llama32-3b-kpath-partial-fullname" --output_path "results/llama32_3b_kpath_partial_fullname/" --prompt_type "raw" --shuffle_num 1 > "./log/llama32_3b_tune_raw2.log" 2>&1 &
