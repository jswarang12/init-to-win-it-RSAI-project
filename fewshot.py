import os
import json
import torch
import random
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    pipeline
)

DATASET_PATH = "dataset/numGLUE/Type_1"
MODELS = ["Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct"]

OUTPUT_DIR = "qwen_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 🔹 Load Dataset
# ==========================
def load_datasets(folder_path):
    datasets = {}
    few_shot_examples = []
    training_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            subject = filename.replace(".json", "").replace("_", " ")
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)                
                for item in data:
                    item["answer"] = str(item["answer"])

                few_shot_examples.extend(data[:50]) 
                training_data.extend(data[50:])

                datasets[subject] = Dataset.from_list(data)
                print(f"Loaded {len(data)} examples for subject: {subject}")

            except json.JSONDecodeError as e:
                print(f"Error loading {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error in {filename}: {e}")

    print(f"\n Total subjects loaded: {len(datasets)}")
    return datasets, few_shot_examples, training_data

dataset, few_shot_examples, training_data = load_datasets(DATASET_PATH)
training_dataset = Dataset.from_list(training_data)

# ==========================
# 🔹 Format Data for Qwen2 (ChatML)
# ==========================
def format_example(example):
    return { 
        "input_text": f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n{example['answer']}<|im_end|>"
    }
formatted_dataset = training_dataset.map(format_example)

# ==========================
# 🔹 Tokenization
# ==========================
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# ==========================
# 🔹 Fine-Tuning Function
# ==========================

# Split training dataset into train (90%) and eval (10%)
split_ratio = 0.9
split_idx = int(len(training_data) * split_ratio)

train_data = training_data[:split_idx]
eval_data = training_data[split_idx:]

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

def fine_tune(model_name):
    print(f"\n🚀 Training {model_name}...\n")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize dataset
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True,
    )

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        padding=True
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{model_name.split('/')[-1]}",
        evaluation_strategy="epoch",
        # evaluation_strategy="no",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        remove_unused_columns=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train Model
    trainer.train()

    # Save Model
    model.save_pretrained(f"{OUTPUT_DIR}/{model_name.split('/')[-1]}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/{model_name.split('/')[-1]}")
    print(f"\n Model {model_name} fine-tuned and saved!\n")

# ==========================
# 🔹 Train Both Models
# ==========================
for model_name in MODELS:
    fine_tune(model_name)

# ==========================
# 🔹 Few-Shot Inference with Retrieval
# ==========================
def retrieve_few_shot_examples(question, k=3):
    """Retrieve `k` most similar few-shot examples (random selection for now)."""
    return random.sample(few_shot_examples, min(len(few_shot_examples), k))

def test_model(model_path):
    print(f"\n🔍 Testing {model_path} with few-shot retrieval...\n")
    
    pipe = pipeline("text-generation", model=model_path)

    # New Question
    new_question = "How many dimes equal $9?"

    # Retrieve Few-Shot Examples
    retrieved_examples = retrieve_few_shot_examples(new_question, k=3)

    # Format Context
    context = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in retrieved_examples])

    # Construct Few-Shot Prompt
    prompt = f"{context}\nQ: {new_question}\nA: "

    # Run Model
    response = pipe(prompt, max_length=50)
    
    print(f"📝 Model Response: {response}\n")

# ==========================
# 🔹 Run Inference on Both Models
# ==========================
for model in MODELS:
    test_model(f"{OUTPUT_DIR}/{model.split('/')[-1]}")
