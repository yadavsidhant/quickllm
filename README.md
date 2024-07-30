# QuickLLM: Fast and Easy Fine-tuning for Popular Language Models

QuickLLM is a Python library that simplifies the process of fine-tuning and interacting with popular language models. With QuickLLM, you can quickly adapt pre-trained models to your specific tasks and chat with them effortlessly.

## Features

- Fine-tune popular language models on custom datasets
- Optimize models for specific tasks like chat, code generation, or domain-specific conversations
- Visualize model parameters and training progress
- Interactive chat interface with fine-tuned models
- Support for a wide range of popular language models

## Installation

You can install QuickLLM using pip:

```bash
pip install quickllm
```

## Quick Start

Here's a simple example to get you started with QuickLLM:

```python
from quickllm import QuickLLM

# Initialize QuickLLM
llm = QuickLLM(model_name="gpt2", input_file="path/to/your/data.csv", output_dir="path/to/output")

# Fine-tune the model
llm.finetune(objective="chat", epochs=3, learning_rate=2e-5)

# Chat with the fine-tuned model
response = llm.chat("Hello, how are you?")
print(response)

# Visualize the model
llm.visualize()

# Start an interactive chat session
from quickllm.chat import start_chat_interface
start_chat_interface(llm.finetuned_model)
```

## Supported Models

QuickLLM supports a wide range of popular language models. Here's a list of currently available models:

1. GPT Family:
   - gpt2
   - gpt2-medium
   - gpt2-large
   - gpt2-xl

2. LLaMA Family:
   - llama
   - llama2
   - llama2-7b
   - llama2-13b
   - llama2-70b

3. BERT Family:
   - bert-base-uncased
   - bert-large-uncased
   - roberta-base
   - roberta-large

4. T5 Family:
   - t5-small
   - t5-base
   - t5-large

5. BART Family:
   - facebook/bart-base
   - facebook/bart-large

6. GPT-Neo Family:
   - EleutherAI/gpt-neo-125M
   - EleutherAI/gpt-neo-1.3B
   - EleutherAI/gpt-neo-2.7B

7. GPT-J Family:
   - EleutherAI/gpt-j-6B

8. OPT Family:
   - facebook/opt-125m
   - facebook/opt-350m
   - facebook/opt-1.3b

9. BLOOM Family:
   - bigscience/bloom-560m
   - bigscience/bloom-1b1
   - bigscience/bloom-1b7

10. Other Models:
    - microsoft/DialoGPT-medium
    - facebook/blenderbot-400M-distill

You can use any of these models by specifying the model name when initializing QuickLLM. More comming soon

## Fine-tuning Objectives

QuickLLM supports different fine-tuning objectives to optimize the model for specific tasks:

- `chat`: General conversational fine-tuning
- `code`: Optimize for code generation tasks
- `specific_chat`: Fine-tune for domain-specific conversations based on your input data

## Visualization

QuickLLM provides built-in visualization tools to help you understand your fine-tuned model:

- Model architecture visualization
- Parameter size distribution
- Training loss curves (if available)

## Contributing

We welcome contributions to QuickLLM! Please feel free to submit issues, fork the repository and send pull requests!

## License

QuickLLM is released under the MIT License. See the [LICENSE](https://github.com/yadavsidhant/quickllm?tab=MIT-1-ov-file) file for more details.

## Contact

If you have any questions, feel free to reach out to me at [supersidhant10@gmail.com](mailto:supersidhant10@gmail.com) or open an issue on our GitHub repository.

Happy fine-tuning!
