import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_training():
    log_path = "results/training_log.csv"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found. Run training first.")
        return

    # Read CSV
    try:
        df = pd.read_csv(log_path)
        # Check if header exists
        if 'epoch' not in df.columns:
            print("Warning: CSV header missing. Reloading with manual column names.")
            df = pd.read_csv(log_path, header=None, names=['epoch', 'train_loss', 'val_loss', 'lr'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Plot Loss
    plt.figure(figsize=(10, 6))
    if 'epoch' in df.columns and 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    else:
        print("Error: Could not determine columns for plotting.")
        return
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss\nData Source: results/training_log.csv')
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    os.makedirs("results", exist_ok=True)
    plot_path = "results/loss_curve.png"
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")

if __name__ == "__main__":
    visualize_training()
