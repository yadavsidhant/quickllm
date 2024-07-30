import pandas as pd
from datasets import Dataset

def load_dataset(input_file):
    # Assume the input file is a CSV with 'text' column
    df = pd.read_csv(input_file)
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.1)

def tokenize_dataset(dataset, tokenizer, objective):
    def tokenize_function(examples):
        if objective == "chat":
            return tokenizer(examples["text"], truncation=True, padding="max_length")
        elif objective == "code":
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        elif objective == "specific_chat":
            # Add any specific preprocessing for the chat based on the text file content
            return tokenizer(examples["text"], truncation=True, padding="max_length")
        else:
            raise ValueError(f"Unsupported objective: {objective}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets