import os
from typing import List, Optional
from .models import SUPPORTED_MODELS
from .finetune import finetune_model
from .chat import chat_with_model
from .visualize import visualize_model

class QuickLLM:
    def __init__(self, model_name: str, input_file: str, output_dir: str):
        if not input_file.endswith('.txt'):
            raise ValueError("Input file must be a .txt file")
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model_name = model_name
        self.input_file = input_file
        self.output_dir = output_dir
        self.finetuned_model = None

    def finetune(self, objective: str, epochs: int = 3, learning_rate: float = 2e-5, 
                 train_split: float = 0.8, validation_split: Optional[float] = None,
                 save_steps: int = 500, eval_steps: int = 500):
        """
        Fine-tune the selected model on the input data.
        
        Args:
            objective (str): The fine-tuning objective (e.g., 'chat', 'code', 'specific_chat')
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for fine-tuning
            train_split (float): Proportion of data to use for training (default: 0.8)
            validation_split (float, optional): Proportion of data to use for validation. 
                                                If None, it will use the remaining data after train_split.
            save_steps (int): Number of steps before saving a checkpoint
            eval_steps (int): Number of steps before running evaluation
        """
        self.finetuned_model = finetune_model(
            model_name=self.model_name,
            input_file=self.input_file,
            output_dir=self.output_dir,
            objective=objective,
            epochs=epochs,
            learning_rate=learning_rate,
            train_split=train_split,
            validation_split=validation_split,
            save_steps=save_steps,
            eval_steps=eval_steps
        )

    def chat(self, message: str) -> str:
        """
        Chat with the fine-tuned model.
        
        Args:
            message (str): User input message
        
        Returns:
            str: Model's response
        """
        if self.finetuned_model is None:
            raise ValueError("Model has not been fine-tuned yet. Call finetune() first.")
        
        return chat_with_model(self.finetuned_model, message)

    def visualize(self):
        """
        Visualize the fine-tuned model's parameters and performance.
        """
        if self.finetuned_model is None:
            raise ValueError("Model has not been fine-tuned yet. Call finetune() first.")
        
        visualize_model(self.finetuned_model, self.output_dir)

    @staticmethod
    def list_supported_models() -> List[str]:
        """
        Get a list of supported model names.
        
        Returns:
            List[str]: List of supported model names
        """
        return list(SUPPORTED_MODELS.keys())
