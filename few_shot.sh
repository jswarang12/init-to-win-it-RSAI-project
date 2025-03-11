#!/bin/bash
# run_all.sh

# Define arrays for few-shot counts and models.
FEW_SHOT_COUNTS=(1 3 5)
MODELS=("meta-llama/llama-3.2-3b-instruct" "meta-llama/llama-3.2-1b-instruct" "deepseek/deepseek-r1-distill-qwen-1.5b" "mistralai/ministral-3b")

# Define indexed arrays for dataset names and corresponding number of questions.
DATASET_NAMES=("Algebra" "Counting_&_Probability" "Geometry" "Intermediate_Algebra" "Number_Theory" "Prealgebra" "Precalculus")
DATASET_COUNTS=(124 38 41 97 62 82 56)

# Loop over all combinations.
for few_shot in "${FEW_SHOT_COUNTS[@]}"; do
    for i in "${!DATASET_NAMES[@]}"; do
        dataset="${DATASET_NAMES[$i]}"
        num_questions="${DATASET_COUNTS[$i]}"
        # Assuming CSV files are stored as: dataset/MATH-500/Test/<Dataset>.csv
        csv_path="dataset/MATH-500/Test/${dataset}.csv"
        for model in "${MODELS[@]}"; do
            echo "Running with FEW_SHOT_COUNT=${few_shot}, DATASET=${dataset}, NUM_QUESTIONS=${num_questions}, CSV_PATH=${csv_path}, MODEL=${model}"
            
            # Export environment variables to override the default config in your Python script.
            export FEW_SHOT_COUNT=$few_shot
            export DATASET_NAME=$dataset
            export NUM_QUESTIONS=$num_questions
            export CSV_PATH=$csv_path
            export MODEL=$model
            
            # Run the Python script (replace wandb_fewshot module name with your script name if needed)
            python -m wandb_fewshot
            
            # Optionally add a sleep delay if needed.
            sleep 1
        done
    done
done
