import torch
import numpy as np
import matplotlib.pyplot as plt
import src.core.config as config
from src.core.physics import tri_layer_model_torch, get_frequencies
from src.core.model import ConditionalDiffusionModel
from src.core.diffusion import sample
import src.core.utils as utils
import os

def get_theoretical_curves():
    """
    Generates theoretical phase curves for Weak Bond (K=1e12) and Strong Bond (K=1e14).
    Returns normalized centered curves ready for the model, along with raw data for plotting/p2p calc.
    """
    print("\n--- 1. PHYSICS REFERENCE: Generating Theoretical Curves ---")
    
    scenarios = {
        "Weak Bond": 1.0e12,
        "Strong Bond": 1.0e14
    }
    
    freqs = get_frequencies()
    results = {}

    # Load stats for normalization
    stats = utils.load_stats(config.DEVICE)
    if stats is None: return None, None
    
    for label, k_val in scenarios.items():
        k_tensor = torch.tensor([k_val]).float()
        
        # 1. Physics Engine Execution
        # Returns raw phase (usually wrapped around pi or similar)
        raw_phase = tri_layer_model_torch(freqs, k_tensor).detach().numpy().flatten()
        
        # 2. Pre-processing (Match generate_data.py)
        # Unwrap
        phase_unwrapped = np.unwrap(np.deg2rad(raw_phase))
        phase_deg = np.rad2deg(phase_unwrapped)
        
        # Normalize: MUST MATCH src/tasks/generate.py EXACTLY
        # generate.py uses: (X_deg - phase_mean_global) / phase_std_global
        
        # 1. Center (Global Mean)
        # Ensure stats are loaded correctly
        if stats and 'phase_mean' in stats:
            phase_mean_global = stats['phase_mean']
            if isinstance(phase_mean_global, torch.Tensor): phase_mean_global = phase_mean_global.item()
            # Use global mean, NOT instance mean
            phase_centered = phase_deg - phase_mean_global
        else:
            print("Warning: 'phase_mean' not found in stats. Using instance mean (incorrect for inference).")
            phase_centered = phase_deg - np.mean(phase_deg)

        # 2. Normalize (Global Std)
        phase_norm = torch.tensor(phase_centered, dtype=torch.float32).to(config.DEVICE)
        
        if stats and 'phase_std' in stats:
            phase_std = stats['phase_std']
            if isinstance(phase_std, torch.Tensor): phase_std = phase_std.item()
            phase_norm = phase_norm / (phase_std + 1e-8)
        else:
             print("Warning: 'phase_std' not found in stats. Using instance std (incorrect for inference).")
             phase_norm = phase_norm / (np.std(phase_centered) + 1e-8)
        
        # Store
        results[label] = {
            "k_true": k_val,
            "raw_phase": raw_phase,
            "phase_centered": phase_centered,
            "model_input": phase_norm.unsqueeze(0).unsqueeze(0) # [1, 1, Points]
        }
        
        print(f"  -> {label} (K={k_val:.1e}): Generated.")

    return results, stats

def run_inference(results, stats):
    """
    Runs the diffusion model on the generated curves.
    """
    print("\n--- 2. AI INFERENCE: Predicting Stiffness ---")
    
    device = config.DEVICE
    model = ConditionalDiffusionModel().to(device)
    
    # Load Model
    try:
        model.load_state_dict(torch.load('checkpoints/model_best.pt', map_location=device, weights_only=True))
        print("  -> Loaded 'model_best.pt'")
    except:
        print("  -> Warning: 'model_best.pt' not found, trying 'model_final.pt'")
        try:
             model.load_state_dict(torch.load('checkpoints/model_final.pt', map_location=device, weights_only=True))
        except:
            print("  -> Error: No checkpoints found.")
            return

    for label, data in results.items():
        phase_input = data['model_input']
        
        # Run Sampling (averaged over a few runs for stability, or just 1 as per request)
        # Using 5 samples to get a robust estimate
        preds_k = []
        # Correctly pass phase_input[0] which is [1, Points] if batch size is 1
        condition_input = phase_input.squeeze(1)

        for _ in range(10):
            pred = sample(model, condition_input, num_samples=1, device=device)
            k_val = utils.inverse_transform_k(pred, stats)
            preds_k.append(k_val)
        
        avg_pred = np.mean(preds_k)
        
        results[label]['k_pred'] = avg_pred
        print(f"  -> {label}: Simulated K={data['k_true']:.2e} | Predicted K={avg_pred:.2e}")

    return results, model # Return model to reuse

def check_consistency(results):
    """
    Analyzes the error and peak-to-peak amplitudes.
    """
    print("\n--- 3. CONSISTENCY CHECK ---")
    
    for label, data in results.items():
        k_true = data['k_true']
        k_pred = data['k_pred']
        phase_centered = data['phase_centered']
        
        # Peak-to-Peak
        p2p = np.max(phase_centered) - np.min(phase_centered)
        
        # Error Calculation
        # Using Log Error since K spans orders of magnitude
        log_error = np.abs(np.log10(k_true) - np.log10(k_pred))
        
        print(f"  [{label}]")
        print(f"    - Simulated K:  {k_true:.2e}")
        print(f"    - Predicted K:  {k_pred:.2e}")
        print(f"    - Self-Consistency Error (Log10 diff): {log_error:.4f}")
        print(f"    - Peak-to-Peak Amplitude: {p2p:.4f} deg")
        
        # Store for analysis
        data['p2p'] = p2p
        data['log_error'] = log_error

def analyze_residuals(results):
    """
    Checks for 'dead zones' if error is high.
    """
    print("\n--- 4. RESIDUAL ANALYSIS ---")
    
    threshold_log_error = 0.5 # Half an order of magnitude
    
    issues_found = False
    for label, data in results.items():
        if data['log_error'] > threshold_log_error:
            issues_found = True
            print(f"  [!] High Error detected for {label} (Log Diff: {data['log_error']:.2f})")
            
            # Check for Dead Zone (Low Amplitude)
            # If amplitude is very small (< 0.5 deg), the signal might be lost in noise/normalization
            if data['p2p'] < 1.0:
                print(f"      -> POSSIBLE DEAD ZONE: Peak-to-Peak amplitude is very low ({data['p2p']:.2f} deg).")
                print(f"      -> The physics parameters (L_BL, Z_ADH, Z_SUB) might be creating a node at these frequencies.")
            else:
                 print(f"      -> Amplitude seems sufficient ({data['p2p']:.2f} deg). Model might be undertrained or OOD.")

    if not issues_found:
        print("  -> No significant anomalies detected. Model is self-consistent within tolerances.")
        
    # Print Physics Params for context
    print("\n  [Context: Physics Parameters]")
    print(f"    - Bondline Thickness (L_BL): {config.L_BL*1e6:.1f} um")
    print(f"    - Adhesive Impedance (Z_ADH): {config.Z_ADH/1e6:.2f} MRayls")
    print(f"    - Substrate Impedance (Z_SUB): {config.Z_SUB/1e6:.2f} MRayls")
    
    # Check Dead Zone Criteria (Quarter Wavelength)
    # Dead zones often occur when L_BL is a multiple of lambda/2 (or lambda/4 depending on boundary)
    # Lambda = c / f
    c_adh = config.C_ADH
    f_center = (config.FREQ_MIN + config.FREQ_MAX) / 2
    wavelength = c_adh / f_center
    
    print(f"    - Center Wavelength in Adhesive: {wavelength*1e6:.1f} um")
    ratio = config.L_BL / wavelength
    print(f"    - L_BL / Lambda ratio: {ratio:.3f}")
    
    if 0.4 < ratio < 0.6:
        print("      -> NOTE: Thickness is near Half-Wavelength (0.5). Resonance expected.")
    elif 0.2 < ratio < 0.3:
        print("      -> NOTE: Thickness is near Quarter-Wavelength (0.25).")

def evaluate_noise_sensitivity(model, stats, device):
    """
    Evaluates how the model's stiffness prediction uncertainty varies with input noise.
    """
    print("\n--- 5. NOISE SENSITIVITY ANALYSIS ---")
    
    # Parameters
    k_true = 1.0e14  # Fixed Ground Truth Stiffness
    noise_levels = np.linspace(0.0, 0.5, 11)  # Sigma levels
    num_samples = 20  # Samples per noise level for distribution
    
    print(f"  -> Target Stiffness: {k_true:.2e} N/m^3")
    print(f"  -> Noise Levels: {noise_levels}")

    # Generate Clean Curve
    freqs = get_frequencies().to(device)
    k_tensor = torch.tensor([k_true]).float().to(device)
    
    # Physics Engine
    raw_phase = tri_layer_model_torch(freqs, k_tensor).detach().cpu().numpy().flatten()
    
    # Pre-process (Unwrap)
    phase_unwrapped = np.unwrap(np.deg2rad(raw_phase))
    phase_deg_clean = np.rad2deg(phase_unwrapped)
    
    # Stats for normalization
    phase_mean = stats['phase_mean']
    phase_std = stats['phase_std']
    if isinstance(phase_mean, torch.Tensor): phase_mean = phase_mean.item()
    if isinstance(phase_std, torch.Tensor): phase_std = phase_std.item()

    # Results storage
    means = []
    cis_lower = []
    cis_upper = []
    
    model.eval()
    
    for sigma in noise_levels:
        preds_k = []
        
        # Run multiple samples to get distribution
        for _ in range(num_samples):
            # Add Noise
            noise = np.random.normal(0, sigma, phase_deg_clean.shape)
            phase_noisy = phase_deg_clean + noise
            
            # Normalize
            phase_centered = phase_noisy - phase_mean
            phase_norm = phase_centered / (phase_std + 1e-8)
            
            # To Tensor [1, Points]
            condition_tensor = torch.tensor(phase_norm, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                 pred_log_k_norm = sample(model, condition_tensor, num_samples=1, device=device)
            
            # Inverse Transform
            pred_k = utils.inverse_transform_k(pred_log_k_norm, stats)
            preds_k.append(pred_k)
            
        # Calculate Stats
        preds_k = np.array(preds_k)
        mean_k = np.mean(preds_k)
        
        # 95% CI
        lower = np.percentile(preds_k, 2.5)
        upper = np.percentile(preds_k, 97.5)
        
        means.append(mean_k)
        cis_lower.append(lower)
        cis_upper.append(upper)
        
        print(f"  -> Sigma {sigma:.2f}: Mean K={mean_k:.2e} [{lower:.2e}, {upper:.2e}]")

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Ground Truth Line
    plt.axhline(y=k_true, color='g', linestyle='-', label=f'Ground Truth (K={k_true:.1e})')
    
    # Predictions
    plt.plot(noise_levels, means, 'b-o', label='Mean Prediction')
    plt.fill_between(noise_levels, cis_lower, cis_upper, color='b', alpha=0.2, label='95% Confidence Interval')
    
    plt.yscale('log')
    plt.ylim(1e13, 1e15)
    plt.xlabel('Input Noise Level (Sigma)')
    plt.ylabel('Predicted Stiffness (N/m^3)')
    plt.title(f'Noise Sensitivity Analysis\nTarget K={k_true:.1e}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    out_path = 'results/noise_sensitivity.png'
    plt.savefig(out_path)
    print(f"Sensitivity plot saved to {out_path}")


def verify_task():
    # 1. Generate
    results, stats = get_theoretical_curves()
    if results is None: return
    
    # 2. Infer
    results, model = run_inference(results, stats)
    
    # 3. Check
    check_consistency(results)
    
    # 4. Analyze
    analyze_residuals(results)
    
    # 5. Noise Sensitivity
    evaluate_noise_sensitivity(model, stats, config.DEVICE)

if __name__ == "__main__":
    verify_task()
