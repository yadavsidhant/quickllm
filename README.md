# QuickLLM: Quick and Easy Fine-tuning for Popular Language Models and Interaction

A Python package called QuickLLM was created to make dealing with, adjusting, and visualizing large language models (LLMs) easier. It provides both novices and experts with an easy-to-use API that lets you quickly train your models on unique text data, communicate with them, see inside of them, and even interact with them through a graphical user interface.

![imgkl](https://github.com/user-attachments/assets/6619ad52-a405-44a1-85ef-8a5e7be3a394)

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

Minimal Example: Fine-Tuning and Chatting with a Model
Here's how you can fine-tune a model on your text data and start chatting with it:

```py
from quickllm import QuickLLM

# Initialize QuickLLM with your desired model and dataset
quick_llm = QuickLLM(model_name="gpt2", input_file="data/train.txt", output_dir="output/")

# Fine-tune the model
quick_llm.finetune(objective="chat")

# Chat with the model
response = quick_llm.chat("Hello, how are you?")
print("Model:", response)
```

Advanced Example: Utilizing All Features
This example demonstrates fine-tuning a model, visualizing its internals, and interacting via a GUI:

```py
from quickllm import QuickLLM

# Initialize QuickLLM with your desired model and dataset
quick_llm = QuickLLM(model_name="gpt2-medium", input_file="data/train.txt", output_dir="output/")

# Fine-tune the model with some custom parameters
quick_llm.finetune(
    objective="chat",              # Objective could be 'chat', 'code', 'specific_chat', etc.
    epochs=5,                      # Number of training epochs
    learning_rate=3e-5,            # Learning rate
    train_split=0.7,               # Train-validation split ratio
    validation_split=0.15,         # Validation split ratio
    save_steps=250,                # Save model every 250 steps
    eval_steps=250,                # Evaluate model every 250 steps
    quantization="4bit",           # or "8bit", or None for no quantization
    resource_utilization=0.8,      # Use 80% of available resources
    optimization_target="balanced" # or "speed" or "accuracy"
)

# Visualize the model's internals and training progress
quick_llm.visualize()

# Start a command-line chat session
response = quick_llm.chat("What's the weather like today?")
print("Model:", response)

# Start the GUI chat interface
quick_llm.start_gui()
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

## Visualization Capabilities
QuickLLM can generate various visualizations, including:

- Model Architecture: Visualize the model's layers and components.
- Parameter Sizes: Bar plots showing the size of each layer's parameters.
- Attention Heads: Distribution of attention heads across model layers.
- Training Metrics: Graphs of training and validation loss, learning rate schedules.
- Token Embeddings: t-SNE plots of token embeddings, annotated with interesting tokens.

Example Visualizations

After fine-tuning, you can visualize your model's architecture, parameters, and training progress with the following command:

```py
quick_llm.visualize()
```
Visualizations are saved in the specified output_dir as PNG files.

## GUI Chat Interface

QuickLLM includes a graphical user interface (GUI) for interacting with your models. To start the GUI:

```py
quick_llm.start_gui()
```
This launches a window where you can load a model, chat with it, and view the chat history.

## Contributing

We welcome contributions to QuickLLM! Please feel free to submit issues, fork the repository and send pull requests!

## License

QuickLLM is released under the MIT License. See the [LICENSE](https://github.com/yadavsidhant/quickllm?tab=MIT-1-ov-file) file for more details.

## Contact

If you have any questions, feel free to reach out to me at [supersidhant10@gmail.com](mailto:supersidhant10@gmail.com) or open an issue on our GitHub repository.

Happy fine-tuning!
