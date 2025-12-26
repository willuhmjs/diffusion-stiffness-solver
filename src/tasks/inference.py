import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import src.core.config as config
import src.core.utils as utils
from src.core.model import ConditionalDiffusionModel
from src.core.diffusion import sample
from src.core.physics import tri_layer_model_torch

def infer_task(loc_path=None, ref_path=None):
    device = config.DEVICE
    
    # 1. Load Stats
    stats = utils.load_stats(device)
    if stats is None:
        print("CRITICAL ERROR: No training statistics found!")
        print("You MUST run 'generate.py' to create the global stats.")
        print("Inference cannot proceed without correct normalization.")
        return

    # 2. Load Data (Default or Custom)
    if loc_path and ref_path:
        print(f"\n--- INVERSE MAPPING: {os.path.basename(loc_path)} ---")
        try:
            df_loc = pd.read_csv(loc_path)
            df_ref = pd.read_csv(ref_path)
        except Exception as e:
            print(f"Error loading CSVs: {e}")
            return
    else:
        print("\n--- INVERSE MAPPING: Default Spec4 ---")
        try:
            df_loc = pd.read_csv('data/raw/Spec4_Loc3_Rep1.csv')
            df_ref = pd.read_csv('data/raw/Spec4_Ref_Rep1.csv')
        except Exception as e:
            print(f"Error loading default CSVs: {e}")
            return

    df_loc.columns = df_loc.columns.str.strip()
    df_ref.columns = df_ref.columns.str.strip()

    # 3. Process
    real_freqs = df_loc['Frequency'].values * 1e6 
    raw_phase_diff = df_loc['Phase'].values - df_ref['Phase'].values
    
    # Centralized Processing
    curve_tensor, curve_centered, target_freqs = utils.process_experimental_data(real_freqs, raw_phase_diff, stats=stats)
    
    # --- CHECK IDENTIFIABILITY ---
    # Phase Peak-to-Peak Check
    # Weak bonds near half-wave resonance can produce near-zero phase response ("dead zone").
    # If the response is too flat, inversion is ill-posed.
    ptp = curve_centered.max() - curve_centered.min()
    print(f"Phase Peak-to-Peak Amplitude: {ptp:.4f} degrees")
    
    # Threshold: 0.5 degrees (tunable based on noise floor)
    # If below this, we declare unresolvable.
    if ptp < 0.5:
        print(f"\n[!] WARNING: Signal amplitude is below resolution threshold ({ptp:.2f} < 0.5 deg).")
        print("    -> Bond is likely very weak and near a spectral node (Dead Zone).")
        print("    -> Inversion cannot be trusted.")
        return

    # Model expects [Batch, 1, Points]
    # For single inference: [1, 1, Points]
    curve_norm = curve_tensor.to(device).unsqueeze(0).unsqueeze(1) 

    # 4. Load Model
    model = ConditionalDiffusionModel().to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device, weights_only=True))
        print(f"Loaded '{config.MODEL_PATH}'")
    except:
        print("Warning: Config model path not found, trying best...")
        try:
             model.load_state_dict(torch.load('checkpoints/model_best.pt', map_location=device, weights_only=True))
        except:
            print("Error: No checkpoints found. Train the model first.")
            return

    # 5. Run Solver
    print("AI is 'thinking' (sampling 50 hypotheses)...")
    preds_k = []
    # We pass [1, Points] to sample if it expects conditional curve
    # Check sample function signature: sample(model, condition_curve, ...)
    # If sample handles batching, we need condition_curve to be [1, Points]
    
    # Correction: sample expects [Batch, Points] for condition_curve
    condition_input = curve_norm.squeeze(1) # [1, Points]

    for _ in range(50):
        pred = sample(model, condition_input, num_samples=1, device=device)
        k_val = utils.inverse_transform_k(pred, stats)
        
        # --- PHYSICAL CLAMPING ---
        # Cap K at K_MAX_PHYS (e.g., 1e16). 
        # Anything above this is physically indistinguishable from a perfect bond.
        k_val = min(k_val, config.K_MAX_PHYS)
        
        preds_k.append(k_val)

    # 6. Results
    preds_k = np.array(preds_k)
    mean_k = preds_k.mean()
    std_k = preds_k.std()

    print(f"\n--- FINAL RESULTS ---")
    if mean_k >= config.K_MAX_PHYS * 0.99:
        print(f"Predicted Stiffness K: ≥ {config.K_MAX_PHYS:.1e} N/m^3 (Perfect Bond Saturation)")
    else:
        print(f"Predicted Stiffness K: {mean_k:.2e} N/m^3")
    print(f"Uncertainty: +/- {std_k:.2e}")

    # 7. Verification Plot
    if loc_path:
        filename = os.path.basename(loc_path)
    else:
        filename = "default_spec4"
        
    save_path = f"results/fit_{filename}.png"
    verify_curve(mean_k, target_freqs, curve_centered, save_path)

def verify_curve(k_val, freqs, real_curve_centered, save_path):
    k_tensor = torch.tensor([k_val]).float()
    f_tensor = torch.tensor(freqs).float()
    
    # Physics Sim
    sim_phase = tri_layer_model_torch(f_tensor, k_tensor).detach().numpy().flatten()
    sim_phase = np.rad2deg(np.unwrap(np.deg2rad(sim_phase)))
    sim_phase_centered = sim_phase - np.mean(sim_phase)

    # Scale Sim to match Real Amplitude for visual comparison
    scaling_factor = (np.max(real_curve_centered) - np.min(real_curve_centered)) / \
                     (np.max(sim_phase_centered) - np.min(sim_phase_centered))
    
    sim_phase_scaled = sim_phase_centered * scaling_factor

    plt.figure(figsize=(10,6))
    plt.plot(freqs/1e6, real_curve_centered, 'b-', label='Real Data')
    plt.plot(freqs/1e6, sim_phase_scaled, 'r--', label=f'AI Pred (K={k_val:.1e}) [Scaled]')
    plt.title(f"Shape Verification (Amplitude Scaled for Visual)")
    plt.xlabel("Freq (MHz)")
    plt.ylabel("Phase Deviation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    print(f"Verification plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, help="Path to Location CSV")
    parser.add_argument("--ref", type=str, help="Path to Reference CSV")
    args = parser.parse_args()
    
    infer_task(args.loc, args.ref)