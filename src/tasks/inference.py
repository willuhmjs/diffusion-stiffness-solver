import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
    ref_name = "Unknown"
    if loc_path and ref_path:
        print(f"\n--- INVERSE MAPPING: {os.path.basename(loc_path)} ---")
        try:
            df_loc = utils.load_file_to_dataframe(loc_path)
            df_ref = utils.load_file_to_dataframe(ref_path)
            ref_name = os.path.basename(ref_path)
        except Exception as e:
            print(f"Error loading Files: {e}")
            return
    else:
        print("\n--- INVERSE MAPPING: Default Spec4 ---")
        try:
            # Check for default files, preferring CSV but falling back if needed (though utility handles extensions, we hardcode paths here)
            # Just use the existing default path string, utils.load_file_to_dataframe handles the loading.
            # However, if those specific files don't exist, we might want to be robust.
            # For now, keeping original default paths but using the new loader.
            # Update: Changed to .dat as that is what exists in the data directory
            # Also added fallback for Ref file since Spec4_Ref_Rep1.dat is missing
            try:
                df_loc = utils.load_file_to_dataframe('data/raw/Spec4_Loc3_Rep1.dat')
            except:
                try:
                    df_loc = utils.load_file_to_dataframe('data/raw/Spec4_Loc3_Rep1.csv')
                except:
                     # Fallback if specific file missing - pick first available matching pattern if possible,
                     # or error gracefully.
                     print("Error: Default Spec4_Loc3_Rep1 file not found (checked .dat and .csv).")
                     return

            try:
                df_ref = utils.load_file_to_dataframe('data/raw/Spec4_Ref_Rep1.dat')
                ref_name = "Spec4_Ref_Rep1.dat"
            except:
                # Fallback to Spec3 Ref if Spec4 Ref is missing
                print("Warning: Spec4_Ref_Rep1.dat not found, trying Spec3_Ref_Rep1.dat as fallback.")
                try:
                    df_ref = utils.load_file_to_dataframe('data/raw/Spec3_Ref_Rep1.dat')
                    ref_name = "Spec3_Ref_Rep1.dat"
                except:
                    print("Error: No suitable Reference file found for default inference.")
                    return

        except Exception as e:
            print(f"Error loading default CSVs: {e}")
            return

    df_loc.columns = df_loc.columns.str.strip()
    df_ref.columns = df_ref.columns.str.strip()

    # 2.5 Lookup Thickness
    # Try to determine specimen and location from filename to look up thickness
    target_filename = os.path.basename(loc_path) if loc_path else "Spec4_Loc3_Rep1.dat"
    
    # Load metadata
    thickness_val = None
    try:
        meta_path = 'data/metadata/specimen_properties.csv'
        if os.path.exists(meta_path):
            df_meta = pd.read_csv(meta_path)
            # Parse filename (e.g. Spec3_Loc1...)
            # We need to extract Specimen number and Location number
            # Assuming format "SpecX_LocY..."
            import re
            match = re.search(r"Spec(\d+)_Loc(\d+)", target_filename, re.IGNORECASE)
            if match:
                s_num = int(match.group(1))
                l_num = int(match.group(2))
                
                # Lookup
                row = df_meta[(df_meta['Specimen'] == s_num) & (df_meta['Location'] == l_num)]
                if not row.empty:
                    # Check for column name variations
                    col_name = 'Thickness_m' if 'Thickness_m' in df_meta.columns else 'Thickness, m'
                    thickness_val = row.iloc[0][col_name]
                    print(f"Found Metadata Thickness for Spec{s_num} Loc{l_num}: {thickness_val*1e6:.1f} um")
                else:
                    print(f"Warning: Spec{s_num} Loc{l_num} not found in metadata.")
            else:
                print(f"Warning: Could not parse Spec/Loc from filename '{target_filename}'")
        else:
            print("Warning: Metadata file not found at", meta_path)
            
    except Exception as e:
        print(f"Warning: Error looking up thickness: {e}")

    # Override config if found
    if thickness_val is not None:
        config.l_bl = thickness_val
        print(f"-> Overriding config.l_bl with specific thickness: {config.l_bl} m")
    else:
        print(f"Error: Could not find specific thickness for {target_filename} in metadata.")
        print("Aborting inference to prevent incorrect physics parameters.")
        return

    # 3. Process
    real_freqs = df_loc['Frequency'].values * 1e6
    # Unwrap each signal individually before subtracting (prevents wrapping artifacts)
    phase_loc_unwrapped = np.unwrap(df_loc['Phase'].values)
    phase_ref_unwrapped = np.unwrap(df_ref['Phase'].values)
    # Sign convention: instruments have opposite sign to TMM, so
    # exp(loc - ref) matches training sim(ref - loc).
    raw_phase_diff = phase_loc_unwrapped - phase_ref_unwrapped
    
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

    # Normalize thickness
    l_bl_mean = stats.get('l_bl_mean', 0.0)
    l_bl_std = stats.get('l_bl_std', 1.0)
    if isinstance(l_bl_mean, torch.Tensor): l_bl_mean = l_bl_mean.item()
    if isinstance(l_bl_std, torch.Tensor): l_bl_std = l_bl_std.item()
    
    l_bl_norm = (thickness_val - l_bl_mean) / (l_bl_std + 1e-8)
    condition_thick = torch.tensor([l_bl_norm], dtype=torch.float32).to(device).unsqueeze(0) # [1, 1]

    for _ in range(50):
        # Pass thickness to sample
        pred = sample(model, condition_input, condition_thick, num_samples=1, device=device)
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
        filename = "default_inference"
        
    save_path = f"results/fit_{filename}.png"
    # Pass resolved thickness to verification
    verify_curve(mean_k, target_freqs, curve_centered, save_path, ref_name, l_bl=thickness_val)

def verify_curve(k_val, freqs, real_curve_centered, save_path, ref_name="Unknown", k_ref=None, l_bl=None):
    k_tensor = torch.tensor([k_val]).float()
    k_ref_bond = torch.tensor([1e16]).float()
    f_tensor = torch.tensor(freqs).float()
    
    # Use specific thickness if provided, else default
    l_bl_val = l_bl if l_bl is not None else config.L_BL

    common_kwargs = dict(
        z_sub=config.Z_SUB,
        l_bl=l_bl_val,
        c_adh=config.C_ADH,
        alpha=config.ALPHA_ADH,
    )

    # Physics Sim: Phase DIFFERENCE (ref - loc), matching training convention
    # Unwrap each individually before subtracting
    phase_loc = tri_layer_model_torch(f_tensor, k_tensor, **common_kwargs).detach().numpy().flatten()
    phase_ref = tri_layer_model_torch(f_tensor, k_ref_bond, **common_kwargs).detach().numpy().flatten()
    loc_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_loc)))
    ref_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_ref)))
    sim_diff = ref_uw - loc_uw
    # Do NOT center per-curve: the experimental overlay (real_curve_centered,
    # from utils.process_experimental_data) preserves its absolute phase level,
    # so the simulated curve must too or the two would be plotted with a
    # constant vertical offset even for a perfect fit.
    sim_phase_centered = sim_diff

    plt.figure(figsize=(10,6))
    plt.plot(freqs/1e6, real_curve_centered, 'b-', label='Real Data')
    plt.plot(freqs/1e6, sim_phase_centered, 'r--', label=f'AI Pred (K={k_val:.1e})')

    # Reference Sim (Derived from Fracture Energy)
    if k_ref is not None:
        k_ref_tensor = torch.tensor([k_ref]).float()
        ref_loc = tri_layer_model_torch(f_tensor, k_ref_tensor, **common_kwargs).detach().numpy().flatten()
        ref_ref = tri_layer_model_torch(f_tensor, k_ref_bond, **common_kwargs).detach().numpy().flatten()
        ref_loc_uw = np.rad2deg(np.unwrap(np.deg2rad(ref_loc)))
        ref_ref_uw = np.rad2deg(np.unwrap(np.deg2rad(ref_ref)))
        ref_diff = ref_ref_uw - ref_loc_uw
        # Not centered, for the same reason as sim_phase_centered above.
        ref_phase_centered = ref_diff
        
        plt.plot(freqs/1e6, ref_phase_centered, 'g-.', label=f'Ref Trend (K={k_ref:.1e})')

    plt.title(f"Verification (Phase Difference)\nReference: {ref_name}")
    plt.xlabel("Freq (MHz)")
    plt.ylabel("Phase Deviation (deg)")
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