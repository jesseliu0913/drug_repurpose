#!/usr/bin/env bash
set -euo pipefail        # stop on first error and catch missing vars

BASE_DIR="/playpen/jesse/drug_repurpose/grpo_startup"
DATA_ROOT="/playpen/jesse/drug_repurpose"

# ────────────────────────────────────────────────────────────────
# 1)  Read .env (if any) and create timestamped output folders
# ────────────────────────────────────────────────────────────────
[[ -f "${BASE_DIR}/.env" ]] && source "${BASE_DIR}/.env"

RUN_TIME=$(date +"%Y%m%d_%H%M")
RESULTS_DIR="${BASE_DIR}/results/${RUN_TIME}"
LOGS_DIR="${RESULTS_DIR}/logs"
MODELS_DIR="${RESULTS_DIR}/models"
mkdir -p "$LOGS_DIR" "$MODELS_DIR"

# ────────────────────────────────────────────────────────────────
# 2)  Hyper‑parameters (same for every job)
# ────────────────────────────────────────────────────────────────
NUM_ITERATIONS=10
NUM_GENERATIONS=4
BATCH_SIZE=12
GRAD_ACCUM=4
LEARNING_RATE=1e-4

USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Models stay in a fixed order so we can map them to GPU 0‑7
MODELS=(
  'JesseLiu/llama32-3b-pagerank-partial-baseline'
  'JesseLiu/llama32-3b-kpath-partial-baseline'
  'JesseLiu/llama32-1b-pagerank-partial-baseline'
  'JesseLiu/llama32-1b-kpath-partial-baseline'
)
# MODELS=(
#   'JesseLiu/llama32-1b-pagerank-partial-baseline'
#   'JesseLiu/llama32-1b-kpath-partial-baseline'
#   'JesseLiu/llama32-1b-kpath-partial-naive'
#   'JesseLiu/llama32-1b-pagerank-partial-naive'
# )

# ────────────────────────────────────────────────────────────────
# 3)  Helper that launches ONE model on the requested GPU
# ────────────────────────────────────────────────────────────────
train_one () {
  local model="$1"          # full HF repo name
  local gpu_id="$2"         # single GPU index e.g. 0
  local model_name output_name csv_dir csv_prefix csv_file output_dir

  model_name=$(basename "$model")
  output_name="${model_name}-grpo"
  $USE_LORA && output_name="${output_name}-lora"

  # ── choose CSV file (exact logic you had) ─────────────────────
  if [[ "$model_name" == *partial* ]]; then
      csv_dir="${DATA_ROOT}/grpo_part_path"
  else
      csv_dir="${DATA_ROOT}/grpo_part_path"
  fi

  if [[ "$model_name" == *kpath* ]]; then
      csv_prefix="k_path"
  else
      csv_prefix="page_rank"
  fi

  if [[ "$model_name" == *naive* ]]; then
      csv_file="${csv_dir}/${csv_prefix}/train_grpo_naive.csv"
  else
      csv_file="${csv_dir}/${csv_prefix}/train_grpo_baseline.csv"
  fi

  echo "==> [GPU ${gpu_id}] Training ${model_name} with ${csv_file}"

  output_dir="${MODELS_DIR}/${output_name}"
  mkdir -p "$output_dir"

  CUDA_VISIBLE_DEVICES=${gpu_id} \
  python "${BASE_DIR}/grpo_train.py" \
          --model_name "$model" \
          --train_csv "$csv_file" \
          --output_dir "$output_dir" \
          --per_device_train_batch_size $BATCH_SIZE \
          --gradient_accumulation_steps $GRAD_ACCUM \
          --num_iterations $NUM_ITERATIONS \
          --num_generations $NUM_GENERATIONS \
          --learning_rate $LEARNING_RATE \
          $( $USE_LORA && echo "--use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT" ) \
          > "${LOGS_DIR}/grpo_${output_name}.log" 2>&1

  # ── push to Hub on success ────────────────────────────────────
  if [ -d "${output_dir}/final_model" ]; then
    local repo="JesseLiu/${model_name}-grpo"
    $USE_LORA && repo="${repo}-lora"

    python "${BASE_DIR}/push_model_grpo.py" \
           --repo_name "$repo" \
           --model_path "${output_dir}/final_model" \
           >> "${LOGS_DIR}/grpo_${output_name}.log" 2>&1
  fi
}

export -f train_one            # make the function visible to subshells
export BASE_DIR DATA_ROOT RESULTS_DIR LOGS_DIR MODELS_DIR            \
       NUM_ITERATIONS NUM_GENERATIONS BATCH_SIZE GRAD_ACCUM           \
       LEARNING_RATE USE_LORA LORA_R LORA_ALPHA LORA_DROPOUT

# ────────────────────────────────────────────────────────────────
# 4)  Fire off all eight jobs in the background, each on one GPU
# ────────────────────────────────────────────────────────────────
# GPU_IDS=(4 5 6 7)   
GPU_IDS=(0 1 4 5)   
pids=()
for idx in "${!MODELS[@]}"; do
  train_one "${MODELS[$idx]}" "${GPU_IDS[$idx]}" &         # idx ∈ 0‑7 doubles as GPU id
  pids+=($!)
done

# ────────────────────────────────────────────────────────────────
# 5)  Block until every job exits; propagate an error if any fail
# ────────────────────────────────────────────────────────────────
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "✅  All eight trainings finished."
