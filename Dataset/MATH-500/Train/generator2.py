import json

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

# Step 2: Create separate JSONL files for each subject
for subject in subjects:
    filename = f"{subject.replace(' ', '_')}.jsonl"  # Replace spaces with underscores for filenames
    with open(filename, "w", encoding="utf-8") as jsonlfile:
        for record in data_by_subject[subject]:
            jsonlfile.write(json.dumps(record) + "\n")  # Write each record in JSONL format

print(f"Created {len(subjects)} JSONL files for different subjects from train.jsonl.")
