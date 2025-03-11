import os
import asyncio
import pandas as pd
import re
import wandb
from model.open import llm_async
from sklearn.metrics import accuracy_score, f1_score

# --- Configuration ---
# Default configuration. These can be overridden by setting environment variables.
default_config = {
    "few_cot_count": 3,           # Number of CoT demonstration examples (choose from 1, 2, 3, or 5)
    "dataset_name": "Algebra",    # Options: Algebra, Counting_&_Probability, Geometry, Intermediate_Algebra, Number_Theory, Prealgebra, Precalculus
    "csv_path": "dataset/MATH-500/Test/Algebra.csv",  # CSV file location for the dataset
    "model": "meta-llama/llama-3.2-3b-instruct"         # Choose from the provided list of models.
}

# Override defaults with environment variables if provided.
default_config["few_cot_count"] = int(os.environ.get("FEW_COT_COUNT", default_config["few_cot_count"]))
default_config["dataset_name"] = os.environ.get("DATASET_NAME", default_config["dataset_name"])
default_config["csv_path"] = os.environ.get("CSV_PATH", default_config["csv_path"])
default_config["model"] = os.environ.get("MODEL", default_config["model"])

# Initialize wandb with a run name that reflects the dataset, CoT count, and model.
run_name = f"{default_config['dataset_name']}_cot{default_config['few_cot_count']}_{default_config['model'].split('/')[-1]}"
wandb.init(project="RSAI-Project", entity="jswarang12-IIIT Hyderabad", config=default_config, name=run_name)
config = wandb.config

#  variables from wandb config.
NUM_COT_EXAMPLES = config.few_cot_count
csv_file_path = config.csv_path

# --- CoT Prompt Building Functions ---
def build_cot_examples(cot_examples):
    """
    Build a string containing chain-of-thought demonstration examples.
    Each example includes the problem, detailed reasoning (solution), and the final answer.
    """
    examples_str = ""
    for i, ex in enumerate(cot_examples):
        examples_str += f"Example {i+1}:\n"
        examples_str += f"Problem: {ex['problem']}\n"
        examples_str += f"Chain of Thought: {ex['solution']}\n"
        examples_str += f"Final Answer: {ex['answer']}\n\n"
    return examples_str

def build_prompt(problem, cot_examples_str):
    """
    Build the prompt that includes the CoT demonstration examples followed by the new problem.
    """
    prompt = (
        "You are a Math Expert. Solve the following problem using a detailed chain-of-thought reasoning process before providing your final answer.\n"
        "Provide your answer in the exact following format and nothing else:\n"
        "Answer: <solution>\n\n"
        "Where <solution> is the final answer with no extra explanation or text.\n"
        "Ensure there is exactly one space after the colon and no additional newlines or characters.\n\n"
        "Follow the exact format below:\n"
        "Final Answer: <final answer>\n\n"
        "Here are some examples:\n\n"
    )
    prompt += cot_examples_str
    prompt += "Now, solve the following problem:\n"
    prompt += f"Problem: {problem}\n"
    prompt += "Chain of Thought:"
    return prompt


async def process_example(example, cot_examples_str):
    """
    Process a single test example by formatting the prompt with CoT examples and invoking async model inference.
    """
    prompt = build_prompt(example['problem'], cot_examples_str)
    response = await llm_async(model=config.model, prompt=prompt)
    # Extract the final answer by searching for the line starting with "Final Answer:"
    match = re.search(r"Final Answer:\s*(.+)$", response.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    return response.strip()

async def run_inference(dataset, cot_examples_str):
    """
    Runs inference over the dataset in parallel using asyncio.
    """
    tasks = [process_example(ex, cot_examples_str) for ex in dataset]
    return await asyncio.gather(*tasks)

def load_dataset_from_csv(csv_file_path):
    """
    Loads the dataset from a CSV file.
    The first NUM_COT_EXAMPLES rows are used as demonstration (CoT) examples,
    and the remaining rows are used as test examples.
    """
    df = pd.read_csv(csv_file_path)
    cot_examples = df.head(NUM_COT_EXAMPLES).to_dict(orient="records")
    test_examples = df.iloc[NUM_COT_EXAMPLES:].to_dict(orient="records")
    return cot_examples, test_examples

def compute_metrics(predictions, references):
    """
    Compute accuracy and F1 score using sklearn metrics.
    """
    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    
    cot_examples, test_dataset = load_dataset_from_csv(csv_file_path)
    print(f"Loaded {len(test_dataset)} test examples from CSV (excluding {len(cot_examples)} CoT demonstration examples).")
    
    cot_examples_str = build_cot_examples(cot_examples)
    
    #  asynchronous inference on the test examples.
    predictions = asyncio.run(run_inference(test_dataset, cot_examples_str))
    true_answers = [example["answer"].strip() for example in test_dataset]
    
    
    metrics = compute_metrics(predictions, true_answers)
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1']}")
    
    # Log metrics to wandb.
    wandb.log(metrics)
    
    output_df = pd.DataFrame({
        "problem": [example["problem"] for example in test_dataset],
        "true_answer": true_answers,
        "model_prediction": predictions
    })
    output_csv = f"model_output_{config.dataset_name}_cot{config.few_cot_count}_{config.model.split('/')[-1]}.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"Model outputs saved to {output_csv}")
    
    wandb.save(output_csv)
    wandb.finish()
