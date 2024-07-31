from datasets import Dataset

def load_dataset(input_file, train_split=0.8, validation_split=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    dataset = Dataset.from_dict({"text": texts})
    
    if validation_split is None:
        # Split into train and test (which will be used as validation)
        splits = dataset.train_test_split(train_size=train_split)
        return {
            'train': splits['train'],
            'validation': splits['test']
        }
    else:
        # Split into train, validation, and test
        test_split = 1 - train_split - validation_split
        splits = dataset.train_test_split(train_size=train_split, test_size=(validation_split + test_split))
        test_valid_split = splits['test'].train_test_split(train_size=validation_split / (validation_split + test_split))
        return {
            'train': splits['train'],
            'validation': test_valid_split['train'],
            'test': test_valid_split['test']
        }

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
