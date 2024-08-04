#!/bin/bash

# Setting strict mode
set -euo pipefail

# Defining directories
LOG_DIR="logs"
EVAL_LOG_FILE="$LOG_DIR/evaluate_all_models.log"
MODEL_DIR="models/saved_models"

# Creating log directory if it doesn't exist
mkdir -p $LOG_DIR

# Starting logging
exec > >(tee -i $EVAL_LOG_FILE)
exec 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting evaluation of all models."

# Function to evaluate model and log the result
evaluate_model() {
    model_name=$1
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Evaluating $model_name model..."
    
    if python -m src.evaluate --model "$model_name"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Successfully evaluated $model_name model."
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Failed to evaluate $model_name model."
    fi
}

# Listing of models to evaluate
models=("cnn" "transformer" "svm" "bayesian" "vision_transformer")

# Evaluating each model
for model in "${models[@]}"; do
    evaluate_model $model
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Evaluation of all models completed successfully."
