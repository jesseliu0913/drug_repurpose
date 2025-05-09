
if [ -f "${BASE_DIR}/.env" ]; then
  source "${BASE_DIR}/.env"
fi

# Base directories
BASE_DIR="/playpen/hongxuan/drug_repurpose/grpo_startup"
LOGS_DIR="${BASE_DIR}/logs"
MODELS_DIR="${BASE_DIR}/models"
mkdir -p $LOGS_DIR $MODELS_DIR


NUM_ITERATIONS=8
NUM_GENERATIONS=6
LEARNING_RATE=1e-5
GPU_IDS="4"
BATCH_SIZE=2
GRAD_ACCUM=4

# Parse command line arguments
MODELS=('JesseLiu/llama32-1b-cold')
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --gpus) shift; GPU_IDS="$1"; shift ;;
    --batch_size) shift; BATCH_SIZE=$1; shift ;;
    --grad_accum) shift; GRAD_ACCUM=$1; shift ;;
    --iterations) shift; NUM_ITERATIONS=$1; shift ;;
    --generations) shift; NUM_GENERATIONS=$1; shift ;;
    --lr) shift; LEARNING_RATE=$1; shift ;;
    *) shift ;;
  esac
done

# Process each model
for MODEL in "${MODELS[@]}"; do
  # model name + grpo
  MODEL_NAME=$(basename "$MODEL")
  OUTPUT_NAME="${MODEL_NAME}-grpo"
  OUTPUT_DIR="${MODELS_DIR}/${OUTPUT_NAME}"
  mkdir -p $OUTPUT_DIR
  
  CUDA_VISIBLE_DEVICES=$GPU_IDS python ${BASE_DIR}/grpo_train.py \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_iterations $NUM_ITERATIONS \
    --num_generations $NUM_GENERATIONS \
    --learning_rate $LEARNING_RATE \
    > "${LOGS_DIR}/grpo_${OUTPUT_NAME}.log" 2>&1
  
  # Upload model if training was successful
  if [ -d "${OUTPUT_DIR}/final_model" ]; then
    python ${BASE_DIR}/push_model_grpo.py \
      --repo_name "JesseLiu/${MODEL_NAME}-grpo" \
      --model_path "${OUTPUT_DIR}/final_model" \
      >> "${LOGS_DIR}/grpo_${OUTPUT_NAME}.log" 2>&1
  fi
done
