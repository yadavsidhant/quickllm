import os
from transformers import Trainer, TrainingArguments
from .models import SUPPORTED_MODELS
from .utils import load_dataset, tokenize_dataset

def finetune_model(model_name, input_file, output_dir, objective, epochs, learning_rate):
    # Load the model and tokenizer
    model_class, tokenizer_class = SUPPORTED_MODELS[model_name]
    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)

    # Load and preprocess the dataset
    dataset = load_dataset(input_file)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, objective)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=learning_rate,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(output_dir, "finetuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "finetuned_model"))

    return model
