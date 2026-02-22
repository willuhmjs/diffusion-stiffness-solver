import torch
import os
import numpy as np
import src.core.config as config
from src.core.physics import generate_dataset

def generate_data_task():
    print("--- GENERATING DATASET (Global Normalization) ---")
    
    # 1. Generate Raw Physics (With Noise)
    print("Simulating Physics Models...")
    # Since generate_dataset is vectorized, we wrap the call or just add a spinner/message.
    # For now, we'll just let it run (it's usually fast on GPU).
    # If we want a progress bar for generation, we'd need to batch it inside generate_dataset,
    # but that's a core change. We will stick to high-level progress.
    X_raw, Y_raw, _, frequencies, L_raw = generate_dataset(n_samples=config.DATASET_SIZE)

    # 2. Pre-Process Phase
    print("Processing Phase Curves...")
    
    # Unwrap and Convert to Degrees (Engine returns wrapped degrees, we ensure unwrapping)
    # Note: physics.py now returns degrees. We need radians for numpy unwrap
    X_numpy = np.deg2rad(X_raw.cpu().numpy())
    X_unwrapped = np.unwrap(X_numpy, axis=1)
    X_deg = torch.from_numpy(np.rad2deg(X_unwrapped)).float().to(X_raw.device)

    # 3. Log-Transform Stiffness
    Y_log = torch.log10(Y_raw)

    # 4. GLOBAL STANDARDIZATION
    X_centered = X_deg
    
    phase_mean = X_centered.mean().item()
    phase_std = X_centered.std().item()
    
    k_mean = Y_log.mean().item()
    k_std = Y_log.std().item()

    # Pre-Process Thickness
    l_bl_mean = L_raw.mean().item()
    l_bl_std = L_raw.std().item()

    # Apply Normalization
    X_final = (X_centered - phase_mean) / (phase_std + 1e-8)
    Y_final = (Y_log - k_mean) / (k_std + 1e-8)
    L_final = (L_raw - l_bl_mean) / (l_bl_std + 1e-8)

    # 5. Save Statistics for Inference
    stats = {
        'k_mean': k_mean,
        'k_std': k_std,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'l_bl_mean': l_bl_mean,
        'l_bl_std': l_bl_std,
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
        'thickness_values': L_final[train_idx],
        'frequencies': frequencies,
        'stats': stats
    }
    
    val_data = {
        'phase_curves': X_final[val_idx],
        'stiffness_values': Y_final[val_idx],
        'thickness_values': L_final[val_idx],
        'frequencies': frequencies,
        'stats': stats
    }

    # 7. Save
    os.makedirs("data/processed", exist_ok=True)
    
    data_path = config.DATA_PATH
    base, ext = os.path.splitext(data_path)
    train_path = f"{base}_train{ext}"
    val_path = f"{base}_val{ext}"
    
    torch.save({'stats': stats}, data_path) # Saving just stats or full data? utils.load_stats expects 'stats' key
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)

    print(f"\nSuccess! Dataset generated.")
    print(f"Train: {len(train_idx)} samples -> {train_path}")
    print(f"Val:   {len(val_idx)} samples -> {val_path}")
    print(f"Stats: Phase Mean={phase_mean:.2f}, Phase Std={phase_std:.2f}")

if __name__ == "__main__":
    generate_data_task()
