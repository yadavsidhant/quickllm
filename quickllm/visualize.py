# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer

def visualize_model(model, output_dir):
    # Set style for all plots
    plt.style.use('seaborn')
    
    # Visualize model architecture
    plt.figure(figsize=(12, 8))
    plt.title("Model Architecture", fontsize=16)
    plt.text(0.5, 0.5, str(model), ha='center', va='center', fontsize=8, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize model parameters
    param_sizes = [p.numel() for p in model.parameters()]
    param_names = [name for name, _ in model.named_parameters()]

    plt.figure(figsize=(12, 8))
    plt.title("Model Parameter Sizes", fontsize=16)
    sns.barplot(x=param_sizes, y=param_names, palette='viridis')
    plt.xscale('log')
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Layer Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_parameters.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize attention heads
    if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        
        plt.figure(figsize=(10, 6))
        plt.title("Attention Heads Distribution", fontsize=16)
        plt.bar(range(num_layers), [num_heads] * num_layers, color='skyblue')
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Number of Attention Heads', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attention_heads.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Visualize training loss (assuming we have logged the loss)
    try:
        log_file = os.path.join(output_dir, "logs", "training_log.csv")
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            plt.figure(figsize=(12, 6))
            plt.title("Training and Validation Loss", fontsize=16)
            plt.plot(df['step'], df['loss'], label='Training Loss')
            if 'eval_loss' in df.columns:
                plt.plot(df['step'], df['eval_loss'], label='Validation Loss')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Learning rate schedule
            if 'learning_rate' in df.columns:
                plt.figure(figsize=(12, 6))
                plt.title("Learning Rate Schedule", fontsize=16)
                plt.plot(df['step'], df['learning_rate'])
                plt.xlabel('Training Step', fontsize=12)
                plt.ylabel('Learning Rate', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=300, bbox_inches='tight')
                plt.close()

    except Exception as e:
        print(f"Could not visualize training metrics: {e}")

    print(f"Visualizations saved in {output_dir}")

# Add this function to visualize token embeddings
def visualize_token_embeddings(model, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    
    # Perform t-SNE to reduce dimensionality
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the reduced embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
    
    # Annotate some interesting tokens
    interesting_tokens = ['hello', 'world', 'ai', 'machine', 'learning']
    for token in interesting_tokens:
        try:
            token_id = tokenizer.encode(token)[0]
            plt.annotate(token, (reduced_embeddings[token_id, 0], reduced_embeddings[token_id, 1]))
        except:
            pass
    
    plt.title("Token Embeddings Visualization (t-SNE)", fontsize=16)
    plt.xlabel("t-SNE dimension 1", fontsize=12)
    plt.ylabel("t-SNE dimension 2", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_embeddings.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Call this function in the visualize_model function
visualize_token_embeddings(model, output_dir)
