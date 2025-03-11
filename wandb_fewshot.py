import os 
import asyncio
import pandas as pd
import re
import wandb
from model.open import llm_async
from sklearn.metrics import accuracy_score, f1_score

# --- Configuration via wandb ---
# Set default parameters. You can override these via wandb's UI or command-line.
default_config = {
    "few_shot_count": 3,       # Choose from 1, 2, 3, or 5
    "dataset_name": "Algebra", # Options: Algebra, Counting_&_Probability, Geometry, Intermediate_Algebra, Number Theory, Prealgebra, Precalculus
    "num_questions": 124,      # For Algebra dataset: 124; update for other datasets accordingly.
    "csv_path": "dataset/MATH-500/Test/Algebra.csv",
    "model": "meta-llama/llama-3.2-3b-instruct"  # Choose from the provided list.
}

default_config["num_questions"] = int(os.environ.get("NUM_QUESTIONS", default_config["num_questions"]))
default_config["dataset_name"] = os.environ.get("DATASET_NAME", default_config["dataset_name"])
default_config["csv_path"] = os.environ.get("CSV_PATH", default_config["csv_path"])
default_config["model"] = os.environ.get("MODEL", default_config["model"])
default_config["few_shot_count"] = int(os.environ.get("FEW_SHOT_COUNT", default_config["few_shot_count"]))

# Initialize wandb run.
run_name = f"{default_config['dataset_name']}_fs{default_config['few_shot_count']}_{default_config['model'].split('/')[-1]}"
wandb.init(project="RSAI-Project", entity="jswarang12-IIIT Hyderabad", config=default_config, name=run_name)
config = wandb.config

# Number of questions to load from the CSV
NUM_QUESTIONS = config.num_questions 
# Number of few-shot examples from the beginning of the CSV
FEW_SHOT_COUNT = config.few_shot_count

def load_dataset_from_csv(csv_file_path, num_questions):
    """
    Loads the dataset from a CSV file and selects the first `num_questions` examples.
    """
    df = pd.read_csv(csv_file_path).head(num_questions)
    return df.to_dict(orient="records")

# dataset using the config csv_path.
csv_file_path = config.csv_path
dataset = load_dataset_from_csv(csv_file_path, NUM_QUESTIONS)
print(f"Loaded {len(dataset)} examples from CSV.")

# FEW_SHOT_COUNT examples .
few_shot_examples = dataset[:FEW_SHOT_COUNT]
evaluation_examples = dataset[FEW_SHOT_COUNT:]

# Build the few-shot prompt .
few_shot_prompt = "\n\n".join(
    [f"Problem: {ex['problem']}\nAnswer: {ex['answer']}" for ex in few_shot_examples]
)

async def process_example(example):
    """
    Process a single evaluation example by formatting the prompt with the few-shot examples,
    invoking the async model inference, and extracting the answer using regex.
    """
    problem_text = example['problem']
    prompt = (
        "You are a Math Expert. Provide your answer in the exact following format and nothing else:\n"
        "Answer: <solution>\n\n"
        "Where <solution> is the final answer with no extra explanation or text.\n"
        "Ensure there is exactly one space after the colon and no additional newlines, spaces or characters.\n\n"
        "Few-shot examples:\n"
        f"{few_shot_prompt}\n\n"
        f"Problem: {problem_text}\n\n"
        "Answer: "
    )
    
    # Use the model specified in the wandb config.
    response = await llm_async(model=config.model, prompt=prompt)
    # Extract answer using regex; matches line starting with "Answer:"
    match = re.search(r"^Answer:\s*(.+)$", response.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    return response.strip()

async def run_inference(dataset):
    """
    Runs inference over the evaluation dataset in parallel using asyncio.
    """
    tasks = [process_example(ex) for ex in dataset]
    return await asyncio.gather(*tasks)

def compute_metrics(predictions, references):
    """
    Compute accuracy and F1 score using sklearn metrics.
    """
    
    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
    #  asynchronous inference on the evaluation examples
    predictions = asyncio.run(run_inference(evaluation_examples))
    true_answers = [example["answer"].strip() for example in evaluation_examples]
    
    
    metrics = compute_metrics(predictions, true_answers)
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1']}")
    
    # Log metrics to wandb
    wandb.log(metrics)
    
   
    output_df = pd.DataFrame({
        "problem": [example["problem"] for example in evaluation_examples],
        "true_answer": true_answers,
        "model_prediction": predictions
    })
    
    
    output_csv = f"model_output_{config.dataset_name}_fs{config.few_shot_count}_{config.model.split('/')[-1]}.csv"
    output_df.to_csv(output_csv, index=False)
    
    print(f"Model outputs saved to {output_csv}")
    
    wandb.save(output_csv)
    wandb.finish()
