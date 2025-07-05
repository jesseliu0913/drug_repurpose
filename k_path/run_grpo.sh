#!/usr/bin/env bash
# GPUS="6 7" MODELS="llama32-1b-instruct-finalddinter llama32-3b-instruct-finalddinter" DATA_TYPE="ddinter" DATASET="Tassy24/K-Paths-inductive-reasoning-ddinter" ./run_grpo.sh 
# GPUS="7" MODELS="llama32-3b-instruct-finalpharmaDB" DATA_TYPE="pharmaDB" DATASET="Tassy24/K-Paths-inductive-reasoning-pharmaDB" ./run_grpo.sh
# GPUS="5" MODELS="llama32-3b-instruct-finalddinter" DATA_TYPE="ddinter" DATASET="Tassy24/K-Paths-inductive-reasoning-ddinter" ./run_grpo.sh
set -euo pipefail


GPUS=${GPUS:-"0 1 2"}                                           
MODELS=${MODELS:-"meta-llama/Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct Qwen/Qwen1.5-7B"} 
MODEL_DIR="/playpen/jesse/drug_repurpose/k_path/model_weights"  
DATASET="Tassy24/K-Paths-inductive-reasoning-ddinter"           # HF dataset
DATA_TYPE="ddinter"                                             # ddinter | drugbank | pharmaDB
COMMON_FLAGS="--use_lora --batch_size 4 --generations 4 --iterations 1 \
              --lr 1e-5 --beta 0.05 --clip_eps 0.2"        
PY_SCRIPT="grpo_train.py"                                 
LOG_DIR="logs"                                           
OUT_DIR="grpo_runs"                                            

read -ra gpu_arr   <<< "$GPUS"
read -ra model_arr <<< "$MODELS"
if [[ ${#gpu_arr[@]} -ne ${#model_arr[@]} ]]; then
  echo "Error: GPUS and MODELS must have the same count." >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  mdl="${model_arr[$idx]}"
  model_path="${MODEL_DIR}/${mdl}"
  short_name="${mdl//\//_}"

  echo "▶︎ Launching $mdl on GPU $gpu …"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=4
    export WANDB__SERVICE_WAIT=300

    python -u "$PY_SCRIPT" \
        --model_name "$model_path" \
        --dataset "$DATASET" \
        --data_type "$DATA_TYPE" \
        --output_dir "$OUT_DIR/$short_name" \
        $COMMON_FLAGS
  ) >"$LOG_DIR/${short_name}.log" 2>&1 &

done

echo "All jobs started. Tail logs with:"
echo "  tail -f $LOG_DIR/*.log"

wait  
