# finetune.py
import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from .models import SUPPORTED_MODELS
from .utils import load_dataset, tokenize_dataset

def finetune_model(model_name, input_file, output_dir, objective, epochs, learning_rate,
                   train_split, validation_split, save_steps, eval_steps):
    # Load the model and tokenizer
    model_class, tokenizer_class = SUPPORTED_MODELS[model_name]
    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)

    # Set padding token for the model if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load and preprocess the dataset
    dataset = load_dataset(input_file, train_split, validation_split)
    tokenized_dataset = {split: tokenize_dataset(data, tokenizer, objective) 
                         for split, data in dataset.items()}

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # Set to 1 for very small datasets
        per_device_eval_batch_size=1,   # Set to 1 for very small datasets
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=learning_rate,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=1,  # Log every step for small datasets
        do_eval=len(tokenized_dataset['validation']) > 0,  # Only do eval if validation set is not empty
        prediction_loss_only=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if len(tokenized_dataset['validation']) > 0 else None,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(output_dir, "finetuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "finetuned_model"))

    return model
