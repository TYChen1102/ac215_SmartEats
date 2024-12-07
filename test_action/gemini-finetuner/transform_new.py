import json
import random

# File paths
input_file_path = 'train.jsonl'
train_split_path = 'train_split_new.jsonl'
test_split_path = 'test_split_new.jsonl'

# Function to read and parse a JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Read the input data
data = read_jsonl(input_file_path)

# Function to transform the data into the required format
def transform_data(data):
    transformed_data = []
    for entry in data:
        transformed_entry = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{entry['instruction']} {json.dumps(entry['input'])}"}
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": entry['output']}
                    ]
                }
            ]
        }
        transformed_data.append(transformed_entry)
    return transformed_data

# Transform the data
transformed_data = transform_data(data)

# Function to split data into train and test sets
def split_data(data, train_ratio=0.8):
    random.shuffle(data)  # Ensure randomness
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

# Split the transformed data
train_data, test_data = split_data(transformed_data)

# Save the train and test data into JSONL files
def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

# Save train and test splits
save_jsonl(train_data, train_split_path)
save_jsonl(test_data, test_split_path)

print(f"Training data saved to: {train_split_path}")
print(f"Test data saved to: {test_split_path}")
