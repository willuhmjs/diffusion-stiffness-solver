import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import src.core.config as config
from src.core.model import ConditionalDiffusionModel
from src.core.diffusion import add_noise
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_path = "data/processed/train.pt"
    val_path = "data/processed/val.pt"
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: Data not found. Run generate_data.py first.")
        return

    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    
    train_phase = train_data['phase_curves'].to(device)
    train_stiffness = train_data['stiffness_values'].to(device)
    
    val_phase = val_data['phase_curves'].to(device)
    val_stiffness = val_data['stiffness_values'].to(device)
    
    print(f"Training on {len(train_phase)} samples, Validating on {len(val_phase)} samples...")

    # 2. Model & Optimizer
    model = ConditionalDiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    loss_fn = nn.MSELoss()
    
    # Setup Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/model_best.pt"
    final_path = "checkpoints/model_final.pt"
    
    # Setup Logging
    os.makedirs("results", exist_ok=True)
    log_path = "results/training_log.csv"
    
    # Initialize Log File
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
    
    min_val_loss = float('inf')
    
    # 3. Training Loop
    # Strategy: Save best model based on VALIDATION loss.
    # Also log metrics for visualization.
    
    for epoch in range(config.EPOCHS):
        # --- TRAIN ---
        model.train()
        
        # Batch Sampling (Random)
        idx = torch.randint(0, len(train_phase), (config.BATCH_SIZE,))
        real_phase = train_phase[idx]
        real_stiffness = train_stiffness[idx] 
        
        # Diffusion Process
        t = torch.randint(0, config.TIMESTEPS, (config.BATCH_SIZE,), device=device)
        noisy_stiffness, noise = add_noise(real_stiffness, t, device=device)
        
        # Predict Noise
        t_norm = t.view(-1, 1).float() / config.TIMESTEPS
        predicted_noise = model(noisy_stiffness, t_norm, real_phase)
        
        # Calculate Loss
        loss = loss_fn(noise, predicted_noise)
        train_loss = loss.item()
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- VALIDATION ---
        # Evaluate on random batch or full set? 
        # For speed, let's use a larger random batch or iterate full val set occasionally.
        # Given small dataset size, we can probably do a quick pass over a batch or two.
        # Let's use a fixed validation batch for consistency, or just random batch.
        
        model.eval()
        with torch.no_grad():
            # Random Val Batch
            val_idx_batch = torch.randint(0, len(val_phase), (config.BATCH_SIZE,))
            val_phase_batch = val_phase[val_idx_batch]
            val_stiffness_batch = val_stiffness[val_idx_batch]
            
            t_val = torch.randint(0, config.TIMESTEPS, (config.BATCH_SIZE,), device=device)
            noisy_stiffness_val, noise_val = add_noise(val_stiffness_batch, t_val, device=device)
            t_norm_val = t_val.view(-1, 1).float() / config.TIMESTEPS
            
            predicted_noise_val = model(noisy_stiffness_val, t_norm_val, val_phase_batch)
            val_loss_tensor = loss_fn(noise_val, predicted_noise_val)
            val_loss = val_loss_tensor.item()
        
        # Step Scheduler (monitor validation loss)
        scheduler.step(val_loss)
        lr_curr = optimizer.param_groups[0]['lr']
        
        # Save Best Model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            
        # Logging
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr_curr])
            
        # Console Output
        if epoch % 100 == 0:
            print(f"Epoch {epoch:<4} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best Val: {min_val_loss:.5f} | LR: {lr_curr:.2e}")

    # Save Final
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete.")
    print(f"Best Validation Loss: {min_val_loss:.5f} (Saved to {best_path})")
    print(f"Log saved to {log_path}")

if __name__ == "__main__":
    train_model()
