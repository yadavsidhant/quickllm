from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_models

def get_supported_models():
    # List all causal language models available on Hugging Face
    models = list_models(filter="text-generation")
    return [model.modelId for model in models]

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token for the model if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

# This dictionary is no longer needed, but we'll keep a small list of recommended models
RECOMMENDED_MODELS = [
    "gpt2",
    "aaditya/Llama3-OpenBioLLM-8B",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-j-6B",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "microsoft/DialoGPT-medium",
]
