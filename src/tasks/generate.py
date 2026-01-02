import torch
import os
import numpy as np
import src.core.config as config
from src.core.physics import generate_dataset

def generate_data_task():
    print("--- GENERATING DATASET (Global Normalization) ---")
    
    # 1. Generate Raw Physics (With Noise)
    X_raw, Y_raw, _, frequencies = generate_dataset(n_samples=config.DATASET_SIZE)

    # 2. Pre-Process Phase
    print("Processing Phase Curves...")
    
    # Unwrap and Convert to Degrees (Engine returns wrapped degrees, we ensure unwrapping)
    # Note: physics.py now returns degrees. We need radians for numpy unwrap
    X_numpy = np.deg2rad(X_raw.numpy())
    X_unwrapped = np.unwrap(X_numpy, axis=1)
    X_deg = torch.from_numpy(np.rad2deg(X_unwrapped)).float()

    # 3. Log-Transform Stiffness
    Y_log = torch.log10(Y_raw)

    # 4. GLOBAL STANDARDIZATION (Replaces Instance Norm)
    # We calculate Mean and Std across the ENTIRE dataset to preserve relative amplitudes.
    # This ensures weak bonds (low amplitude) look different from strong bonds (high amplitude).
    
    # Calculate Global Stats
    phase_mean = X_deg.mean().item()
    phase_std = X_deg.std().item()
    
    k_mean = Y_log.mean().item()
    k_std = Y_log.std().item()
    
    # Apply Normalization
    X_final = (X_deg - phase_mean) / (phase_std + 1e-8)
    Y_final = (Y_log - k_mean) / (k_std + 1e-8)

    # 5. Save Statistics for Inference
    stats = {
        'k_mean': k_mean,
        'k_std': k_std,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'normalization': 'global_standard'
    }

    # 6. Split Train/Val (80/20)
    print("Splitting Dataset (80% Train, 20% Val)...")
    n_total = len(X_final)
    n_train = int(0.8 * n_total)
    
    # Random permutation
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_data = {
        'phase_curves': X_final[train_idx],
        'stiffness_values': Y_final[train_idx],
        'frequencies': frequencies,
        'stats': stats
    }
    
    val_data = {
        'phase_curves': X_final[val_idx],
        'stiffness_values': Y_final[val_idx],
        'frequencies': frequencies,
        'stats': stats
    }

    # 7. Save
    os.makedirs("data/processed", exist_ok=True)
    
    train_path = "data/processed/train.pt"
    val_path = "data/processed/val.pt"
    data_path = config.DATA_PATH # Save the full dataset stats too for utils.load_stats to find
    
    # Keeping original full save for backward compatibility if needed, or remove it?
    # Task says "split the generated data". We'll save train.pt and val.pt.
    # We can also keep training_data.pt if config relies on it, but config likely points to one file.
    # We'll update train.py to look for these.
    
    torch.save({'stats': stats}, data_path) # Saving just stats or full data? utils.load_stats expects 'stats' key
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)

    print(f"\nSuccess! Dataset generated.")
    print(f"Train: {len(train_idx)} samples -> {train_path}")
    print(f"Val:   {len(val_idx)} samples -> {val_path}")
    print(f"Stats: Phase Mean={phase_mean:.2f}, Phase Std={phase_std:.2f}")

if __name__ == "__main__":
    generate_data_task()
