from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_MODELS = {
    # GPT family
    "gpt2": (AutoModelForCausalLM, AutoTokenizer),
    "gpt2-medium": (AutoModelForCausalLM, AutoTokenizer),
    "gpt2-large": (AutoModelForCausalLM, AutoTokenizer),
    "gpt2-xl": (AutoModelForCausalLM, AutoTokenizer),
    
    # LLaMA family
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "llama2": (AutoModelForCausalLM, AutoTokenizer),
    "llama2-7b": (AutoModelForCausalLM, AutoTokenizer),
    "llama2-13b": (AutoModelForCausalLM, AutoTokenizer),
    "llama2-70b": (AutoModelForCausalLM, AutoTokenizer),
    
    # BERT family
    "bert-base-uncased": (AutoModelForCausalLM, AutoTokenizer),
    "bert-large-uncased": (AutoModelForCausalLM, AutoTokenizer),
    "roberta-base": (AutoModelForCausalLM, AutoTokenizer),
    "roberta-large": (AutoModelForCausalLM, AutoTokenizer),
    
    # T5 family
    "t5-small": (AutoModelForCausalLM, AutoTokenizer),
    "t5-base": (AutoModelForCausalLM, AutoTokenizer),
    "t5-large": (AutoModelForCausalLM, AutoTokenizer),
    
    # BART family
    "facebook/bart-base": (AutoModelForCausalLM, AutoTokenizer),
    "facebook/bart-large": (AutoModelForCausalLM, AutoTokenizer),
    
    # GPT-Neo family
    "EleutherAI/gpt-neo-125M": (AutoModelForCausalLM, AutoTokenizer),
    "EleutherAI/gpt-neo-1.3B": (AutoModelForCausalLM, AutoTokenizer),
    "EleutherAI/gpt-neo-2.7B": (AutoModelForCausalLM, AutoTokenizer),
    
    # GPT-J family
    "EleutherAI/gpt-j-6B": (AutoModelForCausalLM, AutoTokenizer),
    
    # OPT family
    "facebook/opt-125m": (AutoModelForCausalLM, AutoTokenizer),
    "facebook/opt-350m": (AutoModelForCausalLM, AutoTokenizer),
    "facebook/opt-1.3b": (AutoModelForCausalLM, AutoTokenizer),
    
    # BLOOM family
    "bigscience/bloom-560m": (AutoModelForCausalLM, AutoTokenizer),
    "bigscience/bloom-1b1": (AutoModelForCausalLM, AutoTokenizer),
    "bigscience/bloom-1b7": (AutoModelForCausalLM, AutoTokenizer),
    
    # Other models
    "microsoft/DialoGPT-medium": (AutoModelForCausalLM, AutoTokenizer),
    "facebook/blenderbot-400M-distill": (AutoModelForCausalLM, AutoTokenizer),
}