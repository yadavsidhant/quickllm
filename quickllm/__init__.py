import os
import torch
from typing import List, Optional
from .models import SUPPORTED_MODELS
from .finetune import finetune_model
from .chat import chat_with_model
from .visualize import visualize_model
from .gui import ChatInterface

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def finetune(self, objective: str, epochs: int = 3, learning_rate: float = 2e-5, 
                 train_split: float = 0.8, validation_split: Optional[float] = None,
                 save_steps: int = 500, eval_steps: int = 500):
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
            eval_steps=eval_steps,
            device=self.device
        )

    def chat(self, message: str) -> str:
        if self.finetuned_model is None:
            raise ValueError("Model has not been fine-tuned yet. Call finetune() first.")
        
        return chat_with_model(self.finetuned_model, message, self.device)

    def visualize(self):
        if self.finetuned_model is None:
            raise ValueError("Model has not been fine-tuned yet. Call finetune() first.")
        
        visualize_model(self.finetuned_model, self.output_dir)

    def start_gui(self):
        ChatInterface(self.output_dir, self.device)

    @staticmethod
    def list_supported_models() -> List[str]:
        return list(SUPPORTED_MODELS.keys())
