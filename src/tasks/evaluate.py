import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import src.core.config as config
import src.core.utils as utils
from src.core.model import ConditionalDiffusionModel
from src.core.diffusion import sample

def evaluate_parameter_recovery():
    print("\n" + "="*80)
    print("🔬 PARAMETER RECOVERY EVALUATION")
    print("="*80)

    device = config.DEVICE
    
    # 1. Load Data
    # Ensure this matches generate.py output
    val_path = "data/processed/training_data_val.pt"
    if not os.path.exists(val_path):
        # Fallback to old name if new one doesn't exist
        fallback = "data/processed/val.pt"
        if os.path.exists(fallback):
             val_path = fallback
        else:
            # Last resort check for training_data.pt (might contain everything)
            fallback_full = "data/processed/training_data.pt"
            if os.path.exists(fallback_full):
                # Try loading full data and extracting validation part if possible,
                # or just use it but warn user.
                # Ideally generate.py creates split files.
                print(f"Warning: {val_path} not found. Trying full dataset {fallback_full}.")
                val_path = fallback_full
            else:
                print(f"Error: Validation data not found at {val_path} or {fallback}.")
                return
        
    print(f"Loading validation data from: {val_path}")
    val_data = torch.load(val_path, map_location=device, weights_only=False)
    
    # Unpack Data
    # Note: If running on old data without thickness, this will fail or need fallback.
    # Assuming data regeneration as per instructions.
    phase_curves = val_data['phase_curves']
    true_stiffness_norm = val_data['stiffness_values']
    thickness_values = val_data.get('thickness_values')
    
    if thickness_values is None:
        print("Error: Validation data missing thickness values. Please regenerate data.")
        return
        
    stats = val_data['stats']
    
    print(f"Evaluating on {len(phase_curves)} validation samples...")
    
    # 2. Load Model
    model = ConditionalDiffusionModel().to(device)
    checkpoint_path = 'checkpoints/model_best.pt'
    if not os.path.exists(checkpoint_path):
         checkpoint_path = 'checkpoints/model_final.pt'
         
    if not os.path.exists(checkpoint_path):
        print("Error: No model checkpoint found.")
        return

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # 3. Batch Inference
    # We'll process in batches to avoid OOM if val set is large, though here it's likely small.
    batch_size = 50
    preds_norm = []
    
    model.eval()
    num_samples = len(phase_curves)
    
    print("Running inference...")
    # Fix infinite loop if range is wrong or batch_size > num_samples
    num_total = len(phase_curves)
    for i in range(0, num_total, batch_size):
        end_idx = min(i+batch_size, num_total)
        batch_phase = phase_curves[i:end_idx].to(device)
        batch_thick = thickness_values[i:end_idx].to(device)
        current_batch_len = end_idx - i
        
        # We only need 1 sample per input for deterministic evaluation or mean of few samples?
        # Diffusion is stochastic. For robust evaluation, we should ideally sample multiple times and take mean.
        # But for speed in pipeline, we might do 1 sample or small number. 
        # Let's do 1 sample for speed first.
        
        with torch.no_grad():
            # sample returns [Batch, 1]
            # We must pass num_samples = current batch size so the noise vector x matches batch_phase dimensions
            # sample function uses num_samples to determine output size if not broadcasting, 
            # or if broadcasting condition.
            # Here batch_phase is [Batch, 1, 2001]
            # Condition thickness is [Batch, 1]
            # We pass num_samples=current_batch_len
            
            if config.USE_THICKNESS:
                batch_pred = sample(model, batch_phase, batch_thick, num_samples=current_batch_len, device=device)
            else:
                batch_pred = sample(model, batch_phase, None, num_samples=current_batch_len, device=device)
            preds_norm.append(batch_pred.cpu())
            
    preds_norm = torch.cat(preds_norm, dim=0).squeeze()
    
    # 4. Inverse Transform
    # stats contains mean/std for normalization
    # Y_final = (Y_log - k_mean) / k_std
    # Y_log = Y_final * k_std + k_mean
    # K = 10^Y_log
    
    k_mean = stats['k_mean']
    k_std = stats['k_std']
    
    # Move to CPU numpy
    true_norm_np = true_stiffness_norm.cpu().numpy().flatten()
    pred_norm_np = preds_norm.numpy().flatten()
    
    # Ensure scalars are numpy floats for broadcasting
    k_std_val = k_std.item() if isinstance(k_std, torch.Tensor) else k_std
    k_mean_val = k_mean.item() if isinstance(k_mean, torch.Tensor) else k_mean

    true_log = true_norm_np * k_std_val + k_mean_val
    pred_log = pred_norm_np * k_std_val + k_mean_val
    
    true_k = 10**true_log
    pred_k = 10**pred_log
    
    # Filter out negative or zero values if any (though 10**x > 0, this protects against other issues or future changes)
    valid_mask = (true_k > 0) & (pred_k > 0)
    if not np.all(valid_mask):
        print(f"Excluding {np.sum(~valid_mask)} samples with non-positive stiffness values.")
        true_k = true_k[valid_mask]
        pred_k = pred_k[valid_mask]
        true_log = true_log[valid_mask]
        pred_log = pred_log[valid_mask]

    # 5. Metrics
    # RMSE on Log Scale is often more meaningful for stiffness spanning orders of magnitude
    mse_log = np.mean((true_log - pred_log)**2)
    rmse_log = np.sqrt(mse_log)
    
    # RMSE on Linear Scale
    mse_lin = np.mean((true_k - pred_k)**2)
    rmse_lin = np.sqrt(mse_lin)
    
    # Weak Bond RMSE (10^12 - 10^14)
    weak_mask = (true_k >= 1e12) & (true_k <= 1e14)
    if np.sum(weak_mask) > 0:
        weak_true_log = true_log[weak_mask]
        weak_pred_log = pred_log[weak_mask]
        weak_rmse_log = np.sqrt(np.mean((weak_true_log - weak_pred_log)**2))
    else:
        weak_rmse_log = 0.0
        
    print(f"\n--- METRICS ---")
    print(f"Overall RMSE (Log10 K): {rmse_log:.4f}")
    print(f"Weak Bond RMSE (Log10 K): {weak_rmse_log:.4f}")
    
    # 6. Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_k, pred_k, alpha=0.5, s=10, c='blue', label='Predictions')
    
    # Perfect Line
    min_val = 1e10
    max_val = 1e18
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Recovery')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e10, 1e18)
    plt.ylim(1e10, 1e18)
    plt.xlabel('Simulated Stiffness (N/m^3)')
    plt.ylabel('Predicted Stiffness (N/m^3)')
    plt.title(f'Parameter Recovery\nRMSE(log): {rmse_log:.3f}\nReference: Validation Set')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    os.makedirs("results", exist_ok=True)
    save_path = "results/parameter_recovery.png"
    plt.savefig(save_path)
    print(f"Scatter plot saved to {save_path}")
    
    # Return metrics for pipeline report
    return {
        "rmse_log": rmse_log,
        "weak_rmse_log": weak_rmse_log,
        "samples": len(true_k)
    }

if __name__ == "__main__":
    evaluate_parameter_recovery()
