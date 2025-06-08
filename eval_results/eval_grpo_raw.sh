#!/usr/bin/env bash


BASE_RESULTS="/playpen/jesse/drug_repurpose/grpo_startup/results"
# PARENTS=(20250530_1704 20250530_1705)
PARENTS=(20250606_2253)
GPUS=(2 3 0 1)
OUTPUT_ROOT="/playpen/jesse/drug_repurpose/eval_results/results"
LOG_ROOT="/playpen/jesse/drug_repurpose/eval_results/log"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

COUNTER=0

for PARENT in "${PARENTS[@]}"; do
  MODELS_DIR="$BASE_RESULTS/$PARENT/models"

  if [[ ! -d "$MODELS_DIR" ]]; then
    echo "Warning: '$MODELS_DIR' not found, skipping."
    continue
  fi

  echo
  echo "===== DEBUG: Listing '*-lora' under $MODELS_DIR ====="
  ls -d "$MODELS_DIR"/*-lora 2>/dev/null || echo "    → (no matches)"
  echo "====================================================="
  echo

  for LORA_BASE in "$MODELS_DIR"/*-lora; do
    if [[ ! -e "$LORA_BASE" ]]; then
      continue
    fi

    LORA_NAME=$(basename "$LORA_BASE")
    LORA_FINAL="$LORA_BASE/final_model"

    if [[ "$LORA_NAME" == *"1b"* ]]; then
      BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
    else
      BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
    fi

    if [[ "$LORA_NAME" == *"kpath"* ]]; then
      INPUT_FILE="/playpen/jesse/drug_repurpose/grpo_part_path/k_path/train_grpo_baseline.csv"
    else
      INPUT_FILE="/playpen/jesse/drug_repurpose/grpo_part_path/page_rank/train_grpo_baseline.csv"
    fi

    ADAPTER="$LORA_FINAL"
    OUTDIR="$OUTPUT_ROOT/${PARENT}_${LORA_NAME}"
    LOGFILE="$LOG_ROOT/${PARENT}_${LORA_NAME}.log"
    GPU_ID="${GPUS[$(( COUNTER % ${#GPUS[@]} ))]}"

    echo "---------------------------------------------"
    echo "Dispatching job #$((COUNTER+1))"
    echo "  LoRA folder:   $LORA_BASE"
    echo "  Final model:   $ADAPTER"
    echo "  Base model:    $BASE_MODEL"
    echo "  Output dir:    $OUTDIR"
    echo "  Log file:      $LOGFILE"
    echo "  Using GPU:     $GPU_ID"
    echo "---------------------------------------------"
    echo

    mkdir -p "$OUTDIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python eval_model.py \
      --model_name "$BASE_MODEL" \
      --adapter_name "$ADAPTER" \
      --output_path "$OUTDIR" \
      --eval_type "test"\
      --input_file "$INPUT_FILE" \
      --prompt_type "raw" \
      --shuffle_num 1 \
      > "$LOGFILE" 2>&1 &

    (( COUNTER++ ))
  done
done

echo
echo "Dispatched $COUNTER jobs across GPUs: ${GPUS[*]}"
echo "Logs → $LOG_ROOT/"
echo "Outputs → $OUTPUT_ROOT/"
