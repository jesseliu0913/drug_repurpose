#!/usr/bin/env bash

BASE_RESULTS="/playpen/jesse/drug_repurpose/grpo_startup/results"
PARENTS=(20250621_2202 20250621_2229)
GPUS=(4 5 2 3)

OUTPUT_ROOT="/playpen/jesse/drug_repurpose/eval_results/results"
LOG_ROOT="/playpen/jesse/drug_repurpose/eval_results/log"
NODES_FILE="/playpen/jesse/drug_repurpose/PrimeKG/nodes.csv"

# Choose eval type: "test" or "train"
EVAL_TYPE=${EVAL_TYPE:-test}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

COUNTER=0

for PARENT in "${PARENTS[@]}"; do
  MODELS_DIR="$BASE_RESULTS/$PARENT/models"
  [[ ! -d "$MODELS_DIR" ]] && {
    echo "Warning: '$MODELS_DIR' not found, skipping."
    continue
  }

  echo
  echo "===== DEBUG: Listing '*-lora' under $MODELS_DIR ====="
  ls -d "$MODELS_DIR"/*-lora 2>/dev/null || echo "    → (no matches)"
  echo "====================================================="
  echo

  for LORA_BASE in "$MODELS_DIR"/*-lora; do
    [[ ! -e "$LORA_BASE" ]] && continue

    LORA_NAME=$(basename "$LORA_BASE")
    LORA_FINAL="$LORA_BASE/final_model"

    if [[ "$LORA_NAME" == *"1b"* ]]; then
      BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
    else
      BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
    fi

    if [[ "$EVAL_TYPE" == "test" ]]; then
      INPUT_FILE="../split_data/data_analysis/test_data_new.csv"
    else
      if [[ "$LORA_NAME" == *"kpath"* ]]; then
        if [[ "$LORA_NAME" == *"naive"* ]]; then
          INPUT_FILE="/playpen/jesse/drug_repurpose/grpo_part_path/k_path/train_grpo_naive.csv"
        else
          INPUT_FILE="/playpen/jesse/drug_repurpose/grpo_part_path/k_path/train_grpo_baseline.csv"
        fi
      else
        if [[ "$LORA_NAME" == *"naive"* ]]; then
          INPUT_FILE="/playpen/jesse/drug_repurpose/grpo_part_path/page_rank/train_grpo_naive.csv"
        else
          INPUT_FILE="/playpen/jesse/drug_repurpose/grpo_part_path/page_rank/train_grpo_baseline.csv"
        fi
      fi
    fi

    OUTDIR="$OUTPUT_ROOT/${PARENT}_${LORA_NAME}_final_model"
    LOGFILE="$LOG_ROOT/${PARENT}_${LORA_NAME}_final_model.log"
    GPU_ID="${GPUS[$(( COUNTER % ${#GPUS[@]} ))]}"

    echo "---------------------------------------------"
    echo "Job #$((COUNTER+1)): $LORA_NAME on GPU $GPU_ID"
    echo "  Base model:  $BASE_MODEL"
    echo "  Adapter:     $LORA_FINAL"
    echo "  Eval type:   $EVAL_TYPE"
    echo "  Input file:  $INPUT_FILE"
    echo "  Nodes file:  $NODES_FILE"
    echo "  Output dir:  $OUTDIR"
    echo "  Log file:    $LOGFILE"
    echo "---------------------------------------------"
    echo

    mkdir -p "$OUTDIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python eval_model.py \
      --model_name   "$BASE_MODEL" \
      --adapter_name "$LORA_FINAL" \
      --nodes_file   "$NODES_FILE" \
      --input_file   "$INPUT_FILE" \
      --output_path  "$OUTDIR" \
      --prompt_type  "raw" \
      --shuffle_num  1 \
      > "$LOGFILE" 2>&1 &

    (( COUNTER++ ))
  done
done

echo
echo "Dispatched $COUNTER jobs across GPUs: ${GPUS[*]}"
echo "Logs → $LOG_ROOT/"
echo "Outputs → $OUTPUT_ROOT/"
