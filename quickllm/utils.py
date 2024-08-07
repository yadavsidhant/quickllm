import pandas as pd
from datasets import Dataset
from transformers import AutoConfig

def load_dataset(input_file, train_split=0.8, validation_split=None):
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        text_columns = df.select_dtypes(include=['object']).columns
        texts = df[text_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    
    dataset = Dataset.from_dict({"text": texts})
    
    # Handle small datasets
    if len(dataset) < 3:
        return {
            'train': dataset,
            'validation': dataset,
            'test': dataset
        }
    
    if validation_split is None:
        # Split into train and test (which will be used as validation)
        splits = dataset.train_test_split(train_size=max(1, int(len(dataset) * train_split)))
        return {
            'train': splits['train'],
            'validation': splits['test'],
            'test': splits['test']
        }
    else:
        # Split into train, validation, and test
        test_split = 1 - train_split - validation_split
        splits = dataset.train_test_split(train_size=max(1, int(len(dataset) * train_split)), 
                                          test_size=max(1, int(len(dataset) * (validation_split + test_split))))
        test_valid_split = splits['test'].train_test_split(train_size=max(0.5, validation_split / (validation_split + test_split)))
        return {
            'train': splits['train'],
            'validation': test_valid_split['train'],
            'test': test_valid_split['test']
        }

def tokenize_dataset(dataset, tokenizer, objective):
    def tokenize_function(examples):
        # Set padding token if it's not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if objective == "chat":
            return tokenizer(examples["text"], truncation=True, padding="max_length", return_special_tokens_mask=True)
        elif objective == "code":
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_special_tokens_mask=True)
        elif objective == "specific_chat":
            # Add any specific preprocessing for the chat based on the text file content
            return tokenizer(examples["text"], truncation=True, padding="max_length", return_special_tokens_mask=True)
        else:
            raise ValueError(f"Unsupported objective: {objective}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_datasets

def check_model_quantization_support(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name)
        return hasattr(config, 'quantization_config')
    except Exception:
        return False

# Make sure to export all necessary functions
__all__ = ['load_dataset', 'tokenize_dataset', 'check_model_quantization_support']
