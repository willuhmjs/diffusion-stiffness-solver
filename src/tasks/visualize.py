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
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    os.makedirs("results", exist_ok=True)
    plot_path = "results/loss_curve.png"
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")

if __name__ == "__main__":
    visualize_training()
