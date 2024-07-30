import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_model(model, output_dir):
    # Visualize model architecture
    plt.figure(figsize=(12, 8))
    plt.title("Model Architecture")
    plt.text(0.5, 0.5, str(model), ha='center', va='center', fontsize=8)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "model_architecture.png"))
    plt.close()

    # Visualize model parameters
    param_sizes = [p.numel() for p in model.parameters()]
    param_names = [name for name, _ in model.named_parameters()]

    plt.figure(figsize=(12, 8))
    plt.title("Model Parameter Sizes")
    sns.barplot(x=param_sizes, y=param_names)
    plt.xscale('log')
    plt.xlabel('Number of Parameters')
    plt.savefig(os.path.join(output_dir, "model_parameters.png"))
    plt.close()

    # Visualize training loss (assuming we have logged the loss)
    try:
        log_file = os.path.join(output_dir, "logs", "training_log.csv")
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            plt.figure(figsize=(12, 6))
            plt.title("Training Loss")
            plt.plot(df['step'], df['loss'])
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(output_dir, "training_loss.png"))
            plt.close()
    except Exception as e:
        print(f"Could not visualize training loss: {e}")

    print(f"Visualizations saved in {output_dir}")