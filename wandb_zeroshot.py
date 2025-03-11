import os
import asyncio
import pandas as pd
import re
import wandb
from model.open import llm_async

# --- Configuration ---
# Default config parameters; override with environment variables if provided.
default_config = {
    "num_questions": 100,  # Number of questions to process
    "dataset_name": "Algebra",  # Options: Algebra, Counting_&_Probability, Geometry, Intermediate_Algebra, Number_Theory, Prealgebra, Precalculus
    "csv_path": "dataset/MATH-500/Test/Algebra.csv",  # CSV file location for the dataset
    "model": "meta-llama/llama-3.2-3b-instruct"  # Choose from the provided model list
}

default_config["num_questions"] = int(os.environ.get("NUM_QUESTIONS", default_config["num_questions"]))
default_config["dataset_name"] = os.environ.get("DATASET_NAME", default_config["dataset_name"])
default_config["csv_path"] = os.environ.get("CSV_PATH", default_config["csv_path"])
default_config["model"] = os.environ.get("MODEL", default_config["model"])

# Initialize wandb with a run name.
run_name = f"{default_config['dataset_name']}_zeroshot_{default_config['model'].split('/')[-1]}"
wandb.init(project="RSAI-Project", entity="jswarang12-IIIT Hyderabad", config=default_config, name=run_name)
config = wandb.config

NUM_QUESTIONS = config.num_questions
csv_file_path = config.csv_path


prompt_template = (
    "You are a Math Expert. Provide your answer in the exact following format and nothing else:\n"
    "Answer: <solution>\n\n"
    "Where <solution> is the final answer with no extra explanation or text.\n"
    "Ensure there is exactly one space after the colon and no additional newlines or characters.\n\n"
    "Problem: {problem}\n\n"
    "Answer: "
)

async def process_example(example):
    """
    Process a single example by formatting the prompt and invoking model inference.
    """
    prompt = prompt_template.format(problem=example['problem'])
    response = await llm_async(model=config.model, prompt=prompt)
    match = re.search(r"^Answer:\s*(.+)$", response.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    return response.strip()

async def run_inference(dataset):
    """
    Runs inference over the dataset in parallel using asyncio.
    """
    tasks = [process_example(ex) for ex in dataset]
    return await asyncio.gather(*tasks)

def load_dataset_from_csv(csv_file_path, num_questions):
    """
    Loads the dataset from a CSV file and selects the first `num_questions` examples.
    """
    df = pd.read_csv(csv_file_path).head(num_questions)
    return df.to_dict(orient="records")

def compute_metrics(predictions, references):
    """
    Compute accuracy and F1 score using sklearn metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
   
    dataset = load_dataset_from_csv(csv_file_path, NUM_QUESTIONS)
    print(f"Loaded {len(dataset)} examples from CSV.")
    
    #  asynchronous zero-shot inference
    predictions = asyncio.run(run_inference(dataset))
    true_answers = [example["answer"].strip() for example in dataset]
    
   
    metrics = compute_metrics(predictions, true_answers)
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1']}")
    
    # Log metrics to wandb
    wandb.log(metrics)
    
    output_df = pd.DataFrame({
        "problem": [example["problem"] for example in dataset],
        "true_answer": true_answers,
        "model_prediction": predictions
    })
    output_csv = f"model_output_{config.dataset_name}_zeroshot_{config.model.split('/')[-1]}.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"Model outputs saved to {output_csv}")
    
    wandb.save(output_csv)
    wandb.finish()
