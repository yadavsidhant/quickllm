import os
import torch
from transformers import (
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    AutoModelForCausalLM, AutoTokenizer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from .utils import load_dataset, tokenize_dataset

def finetune_model(model_name, input_file, output_dir, objective, epochs, learning_rate,
                   train_split, validation_split, save_steps, eval_steps, device,
                   quantization, resource_utilization, optimization_target):
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token for the model if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply quantization if specified
    if quantization:
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

    # Load and preprocess the dataset
    dataset = load_dataset(input_file, train_split, validation_split)
    tokenized_dataset = {split: tokenize_dataset(data, tokenizer, objective) 
                         for split, data in dataset.items()}

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Determine batch size based on resource utilization
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * resource_utilization
        batch_size = max(1, int(available_memory / (model.config.hidden_size * 4 * 2)))
    else:
        batch_size = 1  # Default to 1 for CPU

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=learning_rate,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        do_eval=len(tokenized_dataset['validation']) > 0,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        optim="adamw_torch",
    )

    # Adjust training parameters based on optimization target
    if optimization_target == "speed":
        training_args.gradient_accumulation_steps = 1
        training_args.warmup_steps = 0
    elif optimization_target == "accuracy":
        training_args.gradient_accumulation_steps = 4
        training_args.warmup_steps = 500
    # "balanced" is the default, so we don't need to change anything

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
