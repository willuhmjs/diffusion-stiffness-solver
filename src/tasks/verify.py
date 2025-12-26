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
        
        # Zero-Center (Global Normalization - Global Mean)
        # Use Global Mean from stats instead of Instance Mean
        if 'phase_mean' in stats:
            phase_mean = stats['phase_mean']
            if isinstance(phase_mean, torch.Tensor): phase_mean = phase_mean.item()
            phase_centered = phase_deg - phase_mean
        else:
            # Fallback (Should not happen if stats loaded correctly)
            print("Warning: phase_mean not in stats. Using instance mean.")
            phase_mean = np.mean(phase_deg)
            phase_centered = phase_deg - phase_mean
        
        # Normalize for Model
        phase_norm = torch.tensor(phase_centered, dtype=torch.float32).to(config.DEVICE)
        
        # Global Normalization Logic
        if 'phase_std' in stats:
            phase_std = stats['phase_std']
            if isinstance(phase_std, torch.Tensor): phase_std = phase_std.item()
            phase_norm = phase_norm / (phase_std + 1e-8)
        else:
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
        print(f"  -> {label}: True K={data['k_true']:.2e} | Predicted K={avg_pred:.2e}")

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
        print(f"    - True K:       {k_true:.2e}")
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

def verify_jefferson_curve(model, stats, device):
    """
    Replicates the Jefferson Lab cure curve tracking test using the core physics engine.
    """
    print("\n--- 5. JEFFERSON LAB REPLICATION TEST ---")
    
    # 1. Define the "True" evolution of parameters (Hypothetical Cure Curve)
    # Time points (arbitrary units, e.g., % cure)
    cure_stages = np.linspace(0, 100, 10)
    
    # True K evolution: 10^14 -> 10^17 (High stiffness regime)
    true_K_values = np.logspace(14, 17, len(cure_stages))
    
    # Velocity evolution: 2150 -> 2630 (Approx range around nominal 2391)
    # Velocity increases with cure
    velocities = np.linspace(2150, 2630, len(cure_stages))

    predicted_K_values = []
    
    # Get Frequencies
    freqs = get_frequencies().to(device)
    
    print(f"{'Stage':<10} | {'True K (N/m^3)':<15} | {'Pred K (N/m^3)':<15} | {'Velocity (m/s)':<15}")
    print("-" * 65)

    model.eval()
    
    with torch.no_grad():
        for i, k_true in enumerate(true_K_values):
            vel = velocities[i]
            
            # Physics Engine Execution using tri_layer_model_torch
            # We assume symmetric boundary conditions (K_top = K_bottom = k_true)
            # And we pass the specific velocity for this cure stage
            k_tensor = torch.tensor([k_true]).float().to(device)
            vel_tensor = torch.tensor([vel]).float().to(device)
            
            # tri_layer_model_torch returns phase in degrees
            raw_phase = tri_layer_model_torch(
                freqs, 
                K_top=k_tensor, 
                K_bottom=k_tensor, # Symmetric
                c_adh=vel_tensor
            )
            
            # Process curve (Unwrap and Normalize)
            # Need to move to CPU for numpy unwrap if needed, but if it's already continuous we might be ok.
            # tri_layer_model_torch output is degrees. 
            # Ideally we unwrap radians.
            raw_phase_np = raw_phase.cpu().numpy().flatten()
            phase_rad = np.deg2rad(raw_phase_np)
            phase_unwrapped = np.unwrap(phase_rad)
            phase_deg = np.rad2deg(phase_unwrapped)
            
            # Center and Normalize
            phase_mean = stats['phase_mean']
            if isinstance(phase_mean, torch.Tensor): phase_mean = phase_mean.item()
            phase_std = stats['phase_std']
            if isinstance(phase_std, torch.Tensor): phase_std = phase_std.item()
            
            phase_centered = phase_deg - phase_mean
            phase_norm = phase_centered / (phase_std + 1e-8)
            
            # Prepare for model [1, Points]
            condition_tensor = torch.tensor(phase_norm, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict
            # Sample returns [Batch, 1]
            # Ensure correct input dimension: sample expects [Batch, Points]
            # phase_norm is a numpy array here from line 255 (implicit broadcast)?
            # Wait, phase_norm calculation above involves numpy scalar/arrays.
            # Convert to tensor properly.
            
            condition_tensor = torch.tensor(phase_norm, dtype=torch.float32).unsqueeze(0).to(device) # [1, Points]
            
            pred_log_k_norm = sample(model, condition_tensor, num_samples=1, device=device)
            
            # Inverse Transform
            pred_k = utils.inverse_transform_k(pred_log_k_norm, stats)
            
            predicted_K_values.append(pred_k)

            print(f"{cure_stages[i]:<10.0f} | {k_true:<15.2e} | {pred_k:<15.2e} | {vel:<15.1f}")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.loglog(cure_stages, true_K_values, 'bo-', label='True Stiffness Evolution')
    plt.loglog(cure_stages, predicted_K_values, 'r--x', label='Predicted Stiffness')
    plt.xlabel('Cure Progress (%)')
    plt.ylabel('Interfacial Stiffness K (N/m^3)')
    plt.title('Stiffness Tracking Validation (Jefferson Lab Parameters)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('results/verify_jefferson_stiffness.png')
    print("Stiffness plot saved to results/verify_jefferson_stiffness.png")
    
    # Resonance Shift Check
    print("\n--- RESONANCE SHIFT CHECK ---")
    k_weak = 1e15
    k_strong = 1e16
    vel_nominal = config.C_ADH # Use nominal from config
    
    k_weak_t = torch.tensor([k_weak]).float().to(device)
    k_strong_t = torch.tensor([k_strong]).float().to(device)
    
    # Use standard tri_layer_model
    # The error "The expanded size of the tensor (-1) isn't allowed" often comes from shape mismatch in broadcasting
    # tri_layer_model_torch expects K_top to have batch dim [Batch, 1] if 1D, or [Batch] if we fix it.
    # Let's ensure they are [1, 1] for a single sample.
    
    if k_weak_t.dim() == 0: k_weak_t = k_weak_t.unsqueeze(0)
    if k_strong_t.dim() == 0: k_strong_t = k_strong_t.unsqueeze(0)
    
    phase_weak = tri_layer_model_torch(freqs, k_weak_t, k_weak_t).detach().cpu().numpy().flatten()
    phase_strong = tri_layer_model_torch(freqs, k_strong_t, k_strong_t).detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs.cpu().numpy() / 1e6, phase_weak, 'b-', label=f'Weak Bond (K={k_weak:.1e})')
    plt.plot(freqs.cpu().numpy() / 1e6, phase_strong, 'r--', label=f'Strong Bond (K={k_strong:.1e})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Phase (deg)')
    plt.title('Resonance Shift Check')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/verify_jefferson_resonance.png')
    print("Resonance plot saved to results/verify_jefferson_resonance.png")


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
    
    # 5. Jefferson Test
    verify_jefferson_curve(model, stats, config.DEVICE)

if __name__ == "__main__":
    verify_task()
