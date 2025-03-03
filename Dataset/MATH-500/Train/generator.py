import json
import csv

# Step 1: Read JSONL file and collect unique subjects
input_file = "train.jsonl"
subjects = set()
data_by_subject = {}

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line.strip())
        subject = record["subject"]
        subjects.add(subject)

        if subject not in data_by_subject:
            data_by_subject[subject] = []
        data_by_subject[subject].append(record)

# Step 2: Create separate CSV files for each subject
for subject in subjects:
    filename = f"{subject.replace(' ', '_')}_train.csv"  # Replace spaces with underscores for filenames
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["problem", "solution", "answer", "level", "unique_id"])
        
        # Write data rows
        for record in data_by_subject[subject]:
            writer.writerow([
                record["problem"],
                record["solution"],
                record["answer"],
                record["level"],
                record["unique_id"]
            ])

print(f"Created {len(subjects)} CSV files for different subjects from train.jsonl.")
