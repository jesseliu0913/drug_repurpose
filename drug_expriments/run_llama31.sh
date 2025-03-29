CUDA_VISIBLE_DEVICES=7 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --output_path "results/llama31_8b/" \
    --prompt_type "fraw" \
    --shuffle_num 1 > ./log/llama31_8b_fraw.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python model_run.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results/llama32_3b/" \
    --prompt_type "fcot" \
    --shuffle_num 1 > ./log/llama32_3b_fcot.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python model_pharmDB.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --output_path "results_pharmDB/llama31_8b/" \
    --prompt_type "cot" \
    --shuffle_num 1 > ./log/llama31_8b_cotpharmDB.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python model_pharmDB.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --output_path "results_pharmDB/llama31_8b/" \
    --prompt_type "raw" \
    --shuffle_num 1 > ./log/llama31_8b_rawpharmDB.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python model_pharmDB.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results_pharmDB/llama32_3b/" \
    --prompt_type "cot" \
    --shuffle_num 1 > ./log/llama32_3b_cotpharmDB.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python model_pharmDB.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_path "results_pharmDB/llama32_3b/" \
    --prompt_type "raw" \
    --shuffle_num 1 > ./log/llama32_3b_rawpharmDB.log 2>&1 &

