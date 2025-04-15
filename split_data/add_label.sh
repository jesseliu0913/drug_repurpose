#!/bin/bash

BASE_DIR="/playpen/jesse/drug_repurpose/split_data/results/gemini"
TEST_DATA_PATH="/playpen/jesse/drug_repurpose/split_data/data_analysis/test_data_new.csv"

PROMPT_TYPES=("cot" "fcot" "raw" "fraw" "gene" "phenotype" "raw3")

for PROMPT in "${PROMPT_TYPES[@]}"; do
    FILE_PATH="${BASE_DIR}/${PROMPT}.jsonl"
    
    echo "Running add_label.py for prompt type: $PROMPT"
    python add_label.py --file_path "$FILE_PATH" --test_data_path "$TEST_DATA_PATH"
done
