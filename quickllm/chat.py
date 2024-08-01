# chat.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from colorama import Fore, Style, init
import os

init(autoreset=True)  # Initialize colorama

def chat_with_model(model, message, max_length=100):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    
    input_ids = tokenizer.encode(message, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def list_saved_models(output_dir):
    saved_models = []
    for root, dirs, files in os.walk(output_dir):
        if 'config.json' in files:
            saved_models.append(os.path.relpath(root, output_dir))
    return saved_models

def select_model(output_dir):
    saved_models = list_saved_models(output_dir)
    
    if not saved_models:
        print(Fore.RED + "No saved models found in the output directory." + Style.RESET_ALL)
        return None

    print(Fore.GREEN + "Available models:" + Style.RESET_ALL)
    for i, model_path in enumerate(saved_models, 1):
        print(f"{i}. {model_path}")
    
    while True:
        try:
            choice = int(input(Fore.YELLOW + "Select a model (enter the number): " + Style.RESET_ALL))
            if 1 <= choice <= len(saved_models):
                return os.path.join(output_dir, saved_models[choice - 1])
            else:
                print(Fore.RED + "Invalid choice. Please try again." + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a number." + Style.RESET_ALL)

def start_chat_interface(output_dir):
    print(Fore.GREEN + Style.BRIGHT + "Welcome to QuickLLM Chat!" + Style.RESET_ALL)
    
    model_path = select_model(output_dir)
    if not model_path:
        return

    print(Fore.GREEN + f"Loading model from {model_path}" + Style.RESET_ALL)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print(Fore.YELLOW + "Type 'exit' to end the conversation." + Style.RESET_ALL)
    
    while True:
        user_input = input(Fore.CYAN + "You: " + Style.RESET_ALL)
        if user_input.lower() == "exit":
            print(Fore.GREEN + Style.BRIGHT + "Thank you for chatting. Goodbye!" + Style.RESET_ALL)
            break
        
        response = chat_with_model(model, user_input)
        print(Fore.MAGENTA + "Model: " + Style.RESET_ALL + response)
        print()  # Add a blank line for better readability

if __name__ == "__main__":
    output_dir = input("Enter the path to your output directory: ")
    start_chat_interface(output_dir)
