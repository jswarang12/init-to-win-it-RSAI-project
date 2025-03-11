#!/bin/bash
# Ensure this file is executed with bash

# Define indexed arrays for dataset names and their corresponding question counts.
datasets=("Algebra" "Counting_&_Probability" "Geometry" "Intermediate_Algebra" "Number_Theory" "Prealgebra" "Precalculus")
counts=(124 38 41 97 62 82 56)

# Define an array of models.
MODELS=("meta-llama/llama-3.2-3b-instruct" "meta-llama/llama-3.2-1b-instruct" "deepseek/deepseek-r1-distill-qwen-1.5b" "mistralai/ministral-3b")

# Loop over each dataset and model combination.
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    num_questions="${counts[$i]}"
    # Construct the CSV path based on the dataset name.
    csv_path="dataset/MATH-500/Test/${dataset}.csv"
    
    for model in "${MODELS[@]}"; do
        echo "Running zero-shot with DATASET=${dataset}, NUM_QUESTIONS=${num_questions}, CSV_PATH=${csv_path}, MODEL=${model}"
        
        # Export environment variables so the Python script uses them.
        export NUM_QUESTIONS=$num_questions
        export DATASET_NAME=$dataset
        export CSV_PATH=$csv_path
        export MODEL=$model
        
        # Run the zero-shot Python script.
        python -m wandb_zeroshot
        
        # Optional: Delay between runs.
        sleep 1
    done
done
