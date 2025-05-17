BASE_DIR="/playpen/hongxuan/Drug/drug_repurpose/grpo_startup"

cd BASE_DIR


DATA_ROOT="/playpen/hongxuan/Drug/drug_repurpose"

# ------------------------------------------------------------------
# .env
# ------------------------------------------------------------------
if [ -f "${BASE_DIR}/.env" ]; then
  source "${BASE_DIR}/.env"
fi

# ------------------------------------------------------------------
# Output folders
# ------------------------------------------------------------------
RUN_TIME=$(date +"%Y%m%d_%H%M")
RESULTS_DIR="${BASE_DIR}/results/${RUN_TIME}"
LOGS_DIR="${RESULTS_DIR}/logs"
MODELS_DIR="${RESULTS_DIR}/models"
mkdir -p "$LOGS_DIR" "$MODELS_DIR"

# ------------------------------------------------------------------
# hyperparameters
# ------------------------------------------------------------------
NUM_ITERATIONS=16
NUM_GENERATIONS=6
BATCH_SIZE=24
GRAD_ACCUM=1
LEARNING_RATE=1e-5
GPU_IDS="6,7"

USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

MODELS=(
  'JesseLiu/llama32-1b-kpath-partial'
  'JesseLiu/llama32-1b-pagerank-partial'
  'JesseLiu/llama32-3b-pagerank-partial'
  'JesseLiu/llama32-3b-kpath-partial'
)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)           shift; MODELS=(); while [[ $# -gt 0 && $1 != --* ]]; do MODELS+=("$1"); shift; done ;;
    --gpus)             shift; GPU_IDS=$1;               shift ;;
    --batch_size)       shift; BATCH_SIZE=$1;            shift ;;
    --grad_accum)       shift; GRAD_ACCUM=$1;            shift ;;
    --iterations)       shift; NUM_ITERATIONS=$1;        shift ;;
    --generations)      shift; NUM_GENERATIONS=$1;       shift ;;
    --lr)               shift; LEARNING_RATE=$1;         shift ;;
    --use_lora)         USE_LORA=true;                   shift ;;
    --lora_r)           shift; LORA_R=$1;                shift ;;
    --lora_alpha)       shift; LORA_ALPHA=$1;            shift ;;
    --lora_dropout)     shift; LORA_DROPOUT=$1;          shift ;;
    *)                  shift ;;
  esac
done

# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL")
  OUTPUT_NAME="${MODEL_NAME}-grpo"
  $USE_LORA && OUTPUT_NAME="${OUTPUT_NAME}-lora"

  # ------ choose CSV file ----------------------------------------------------
  if [[ "$MODEL_NAME" == *partial* ]]; then
      CSV_DIR="${DATA_ROOT}/grpo_part_path"
  else
      CSV_DIR="${DATA_ROOT}/grpo_path"
  fi

  if [[ "$MODEL_NAME" == *kpath* ]]; then
      CSV_FILE="${CSV_DIR}/k_path/train_grpo.csv"
  else
      CSV_FILE="${CSV_DIR}/page_rank/train_grpo.csv"
  fi
  # --------------------------------------------------------------------------
  echo "==> Training ${MODEL_NAME} with CSV file: ${CSV_FILE}"


  OUTPUT_DIR="${MODELS_DIR}/${OUTPUT_NAME}"
  mkdir -p "$OUTPUT_DIR"

  TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} python ${BASE_DIR}/grpo_train.py \
    --model_name ${MODEL} \
    --train_csv ${CSV_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_iterations ${NUM_ITERATIONS} \
    --num_generations ${NUM_GENERATIONS} \
    --learning_rate ${LEARNING_RATE}"

  if $USE_LORA; then
    TRAIN_CMD+=" --use_lora --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT}"
  fi

  echo "==> Training ${MODEL_NAME}"
  eval "${TRAIN_CMD}" > "${LOGS_DIR}/grpo_${OUTPUT_NAME}.log" 2>&1

  # ------ push to Hub if training succeeded ----------------------------------
  if [ -d "${OUTPUT_DIR}/final_model" ]; then
    REPO_NAME="JesseLiu/${MODEL_NAME}-grpo"
    $USE_LORA && REPO_NAME="${REPO_NAME}-lora"

    python ${BASE_DIR}/push_model_grpo.py \
      --repo_name "${REPO_NAME}" \
      --model_path "${OUTPUT_DIR}/final_model" \
      >> "${LOGS_DIR}/grpo_${OUTPUT_NAME}.log" 2>&1
  fi
done
