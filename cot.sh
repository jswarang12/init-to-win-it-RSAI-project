#!/bin/bash
# Ensure this file is executed with bash.

# Define an array of FEW COT COUNT values.
FEW_COT_COUNTS=(1 3 5)

# Define indexed arrays for dataset names and their corresponding number of demonstration+test questions.
datasets=("Algebra" "Counting_&_Probability" "Geometry" "Intermediate_Algebra" "Number_Theory" "Prealgebra" "Precalculus")
# Here the count values represent the total available examples for demonstration purposes.
counts=(124 38 41 97 62 82 56)

# Define an array of models.
MODELS=("meta-llama/llama-3.2-3b-instruct" "meta-llama/llama-3.2-1b-instruct" "deepseek/deepseek-r1-distill-qwen-1.5b" "mistralai/ministral-3b")

# Loop over each FEW COT COUNT.
for cot_count in "${FEW_COT_COUNTS[@]}"; do
    # Loop over each dataset.
    for i in "${!datasets[@]}"; do
        dataset="${datasets[$i]}"
        # The CSV path is built based on the dataset name.
        csv_path="dataset/MATH-500/Test/${dataset}.csv"
        # Note: The count value here (from counts array) can be used to verify dataset size if needed.
        num_available="${counts[$i]}"
        # We'll use a constant number of test questions (e.g., 10) as defined in the Python script.
        num_test_questions=10
        
        for model in "${MODELS[@]}"; do
            echo "Running CoT with FEW_COT_COUNT=${cot_count}, DATASET=${dataset}, CSV_PATH=${csv_path}, MODEL=${model}"
            
            # Export environment variables for the Python script.
            export FEW_COT_COUNT=$cot_count
            export NUM_TEST_QUESTIONS=$num_test_questions
            export DATASET_NAME=$dataset
            export CSV_PATH=$csv_path
            export MODEL=$model
            
            # Run the CoT inference Python script.
            python -m wandb_cot
            
            # Optional: Add a small delay between runs.
            sleep 1
        done
    done
done
