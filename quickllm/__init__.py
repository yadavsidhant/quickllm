import os
import torch
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from .finetune import finetune_model
from .chat import chat_with_model
from .visualize import visualize_model
from .gui import ChatInterface
from .utils import load_dataset, check_model_quantization_support

class QuickLLM:
    def __init__(self, model_name: str, input_file: str, output_dir: str):
        if not input_file.endswith(('.txt', '.csv')):
            raise ValueError("Input file must be a .txt or .csv file")
        
        self.model_name = model_name
        self.input_file = input_file
        self.output_dir = output_dir
        self.finetuned_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def finetune(self, objective: str, epochs: int = 3, learning_rate: float = 2e-5, 
                 train_split: float = 0.8, validation_split: Optional[float] = None,
                 save_steps: int = 500, eval_steps: int = 500, 
                 quantization: Optional[str] = None, resource_utilization: float = 1.0,
                 optimization_target: str = "balanced"):
        
        quantization_supported = check_model_quantization_support(self.model_name)
        
        if quantization and not quantization_supported:
            print(f"Warning: The model {self.model_name} does not support quantization. Proceeding without quantization.")
            quantization = None
        
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
            device=self.device,
            quantization=quantization,
            resource_utilization=resource_utilization,
            optimization_target=optimization_target
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
        # This now returns all models available on Hugging Face
        return ["Any model available on Hugging Face can be used."]

    @staticmethod
    def load_pretrained_model(model_name: str, device: Union[str, torch.device]) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(model_name).to(device)

    @staticmethod
    def load_pretrained_tokenizer(model_name: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(model_name)
