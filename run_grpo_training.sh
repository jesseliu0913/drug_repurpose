#!/bin/bash

# Create the processed data with only questions
echo "Step 1: Processing training data to extract questions only..."
python process_train_data.py

# Run GRPO training with questions-only data
echo "Step 2: Running GRPO training with questions-only data..."
cd grpo_startup
python grpo_train.py --use_questions_only=True

echo "Training completed!" 