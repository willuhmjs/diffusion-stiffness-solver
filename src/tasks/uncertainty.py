import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import src.core.config as config
import src.core.utils as utils
from src.core.model import ConditionalDiffusionModel
from src.core.diffusion import sample
from src.core.physics import tri_layer_model_torch

def uncertainty_task(loc_path=None, ref_path=None, num_samples=100, ax_k=None, ax_phase=None):
    device = config.DEVICE
    if ax_k is None and ax_phase is None:
        print(f"Running Uncertainty Analysis with {num_samples} samples...")
    
    # 1. Load Stats
    stats = utils.load_stats(device)
    if stats is None:
        print("CRITICAL ERROR: No training statistics found!")
        return

    # 2. Load Data (Default or Custom)
    if loc_path and ref_path:
        print(f"\n--- INVERSE MAPPING: {os.path.basename(loc_path)} ---")
        try:
            df_loc = utils.load_file_to_dataframe(loc_path)
            df_ref = utils.load_file_to_dataframe(ref_path)
            filename = os.path.basename(loc_path).replace('.csv', '').replace('.dat', '')
            ref_name = os.path.basename(ref_path)
        except Exception as e:
            print(f"Error loading files: {e}")
            return
    else:
        print("\n--- INVERSE MAPPING: Default Spec4 ---")
        try:
            # Check for .dat files first if .csv files are missing
            loc_default = 'data/raw/Spec4_Loc3_Rep1.csv'
            ref_default = 'data/raw/Spec4_Ref_Rep1.csv'
            
            if not os.path.exists(loc_default):
                loc_default_dat = loc_default.replace('.csv', '.dat')
                if os.path.exists(loc_default_dat):
                    loc_default = loc_default_dat
                else:
                    # Fallback to backup_csv if needed, though this might not be intended
                    pass
            
            if not os.path.exists(ref_default):
                ref_default_dat = ref_default.replace('.csv', '.dat')
                if os.path.exists(ref_default_dat):
                    ref_default = ref_default_dat
                else:
                    # Try finding Spec4_Ref_Rep1.dat explicitly since it might be missing in raw but present as Ref
                    # Actually, based on file list, Spec4_Ref_Rep1.dat does NOT exist in data/raw
                    # But Spec4_Ref_Rep1.csv DOES exist in data/backup_csv
                    pass

            # If raw files are missing, try backup_csv
            if not os.path.exists(loc_default):
                 loc_default = 'data/backup_csv/Spec4_Loc3_Rep1.csv'
            if not os.path.exists(ref_default):
                 # Try finding Spec3 reference if Spec4 reference is missing (common practice in this dataset)
                 ref_default_fallback = 'data/raw/Spec3_Ref_Rep1.dat'
                 if os.path.exists(ref_default_fallback):
                     ref_default = ref_default_fallback
                 else:
                     ref_default = 'data/backup_csv/Spec4_Ref_Rep1.csv'

            print(f"Loading: {loc_default}")
            print(f"Loading: {ref_default}")

            df_loc = utils.load_file_to_dataframe(loc_default)
            df_ref = utils.load_file_to_dataframe(ref_default)
            filename = "default_inference"
            ref_name = os.path.basename(ref_default)
        except Exception as e:
            print(f"Error loading default files: {e}")
            return

    df_loc.columns = df_loc.columns.str.strip()
    df_ref.columns = df_ref.columns.str.strip()

    # 3. Process Data
    real_freqs = df_loc['Frequency'].values * 1e6 
    # Unwrap each signal individually before subtracting (prevents wrapping artifacts)
    phase_loc_unwrapped = np.unwrap(df_loc['Phase'].values)
    phase_ref_unwrapped = np.unwrap(df_ref['Phase'].values)
    # Sign convention: instruments have opposite sign to TMM, so
    # exp(loc - ref) matches training sim(ref - loc).
    raw_phase_diff = phase_loc_unwrapped - phase_ref_unwrapped
    
    curve_tensor, curve_centered, target_freqs = utils.process_experimental_data(real_freqs, raw_phase_diff, stats=stats)
    
    # Check signal strength
    ptp = curve_centered.max() - curve_centered.min()
    if ptp < 0.5:
        print(f"Warning: Low signal amplitude ({ptp:.2f} deg). Results may be unreliable.")

    # Prepare input for model [1, 1, Points] -> [1, Points] for sample function
    curve_norm = curve_tensor.to(device).unsqueeze(0).unsqueeze(1) 
    condition_input = curve_norm.squeeze(1) # [1, Points]

    # --- LOOKUP THICKNESS ---
    target_filename = os.path.basename(loc_path) if loc_path else "Spec4_Loc3_Rep1.dat"
    try:
        thickness_val = config.L_BL 
    except AttributeError:
        # Fallback if config.L_BL is somehow missing, though it should be there
        thickness_val = 0.0002
    
    # Try metadata
    try:
        meta_path = 'data/metadata/specimen_properties.csv'
        if os.path.exists(meta_path):
            df_meta = pd.read_csv(meta_path)
            import re
            match = re.search(r"Spec(\d+)_Loc(\d+)", target_filename, re.IGNORECASE)
            if match:
                s_num = int(match.group(1))
                l_num = int(match.group(2))
                row = df_meta[(df_meta['Specimen'] == s_num) & (df_meta['Location'] == l_num)]
                if not row.empty:
                     col_name = 'Thickness_m' if 'Thickness_m' in df_meta.columns else 'Thickness, m'
                     thickness_val = row.iloc[0][col_name]
                     print(f"Using metadata thickness: {thickness_val*1e6:.1f} um")
    except Exception as e:
        print(f"Thickness lookup failed: {e}")

    l_bl_mean = stats.get('l_bl_mean', 0.0)
    l_bl_std = stats.get('l_bl_std', 1.0)
    if isinstance(l_bl_mean, torch.Tensor): l_bl_mean = l_bl_mean.item()
    if isinstance(l_bl_std, torch.Tensor): l_bl_std = l_bl_std.item()
    
    l_bl_norm = (thickness_val - l_bl_mean) / (l_bl_std + 1e-8)
    condition_thick = torch.tensor([l_bl_norm], dtype=torch.float32).to(device).unsqueeze(0) # [1, 1]

    # 4. Load Model
    model = ConditionalDiffusionModel().to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device, weights_only=True))
    except:
        print("Warning: Config model path not found, trying best...")
        try:
            model.load_state_dict(torch.load('checkpoints/model_best.pt', map_location=device, weights_only=True))
        except:
            print("Error: No checkpoints found.")
            return

    # 5. Sampling Loop
    print("Generating samples...")
    preds_k = []
    
    # We can batch the sampling if the model and GPU memory allow, 
    # but the sample function iterates over timesteps, so for safety we loop or use small batches.
    # The current sample function takes num_samples.
    # Let's try to run all samples in one go if it fits in memory (100 is small), 
    # otherwise we can chunk it.
    
    try:
        # Generate all samples at once
        # Pass thickness
        preds_norm = sample(model, condition_input, condition_thick, num_samples=num_samples, device=device) # [N, 1]
        
        for i in range(num_samples):
            pred_val = preds_norm[i]
            k_val = utils.inverse_transform_k(pred_val, stats)
            k_val = min(k_val, config.K_MAX_PHYS) # Clamp
            preds_k.append(k_val)
            
    except RuntimeError as e:
        print(f"Batch sampling failed (likely OOM), falling back to loop: {e}")
        preds_k = []
        for _ in range(num_samples):
            preds_norm = sample(model, condition_input, condition_thick, num_samples=1, device=device)
            k_val = utils.inverse_transform_k(preds_norm, stats)
            k_val = min(k_val, config.K_MAX_PHYS)
            preds_k.append(k_val)

    preds_k = np.array(preds_k)
    
    # 6. Statistics
    mean_k = np.mean(preds_k)
    std_k = np.std(preds_k)
    median_k = np.median(preds_k)
    ci_lower = np.percentile(preds_k, 2.5)
    ci_upper = np.percentile(preds_k, 97.5)
    
    print(f"\n--- UNCERTAINTY RESULTS ---")
    if mean_k >= config.K_MAX_PHYS * 0.99:
        print(f"Predicted K: ≥ {config.K_MAX_PHYS:.1e} N/m^3 (Perfect Bond Saturation)")
        print("Note: Predictions are clamped at the physical limit of perfect bonding.")
    else:
        print(f"Mean K:   {mean_k:.2e} N/m^3")
    
    print(f"Std Dev:  {std_k:.2e}")
    print(f"95% CI:   [{ci_lower:.2e}, {ci_upper:.2e}]")

    # 7. Visualization - K Distribution
    os.makedirs("results", exist_ok=True)
    
    if ax_k is None:
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
    else:
        ax1 = ax_k

    # Check if we have variance for KDE
    use_kde = std_k > 0 and len(np.unique(preds_k)) > 1
    sns.histplot(preds_k, kde=use_kde, log_scale=True, ax=ax1)
    ax1.axvline(mean_k, color='r', linestyle='--', label=f'Mean: {mean_k:.2e}')
    ax1.axvline(ci_lower, color='g', linestyle=':', label='95% CI')
    ax1.axvline(ci_upper, color='g', linestyle=':')
    ax1.set_xlabel('Stiffness K (N/m^3)')
    
    # Use shorter title for subplots
    title_text = f'{filename}\nMean: {mean_k:.1e}' if ax_k else f'Posterior Distribution of Stiffness K (n={num_samples})\nReference: {ref_name}'
    ax1.set_title(title_text, fontsize=10 if ax_k else 12)
    
    if ax_k is None:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if ax_k is None:
        k_dist_path = f"results/uncertainty_k_dist_{filename}.png"
        plt.savefig(k_dist_path)
        print(f"Saved K distribution to {k_dist_path}")

    # 8. Visualization - Phase Reconstruction
    # Run forward model for all samples (or subset)
    print("Running forward physics for verification...")
    
    k_tensor = torch.tensor(preds_k).float().unsqueeze(1) # [N, 1]
    f_tensor = torch.tensor(target_freqs).float() # [F]

    # Physics Params
    kwargs = dict(
        l_bl=thickness_val,
        z_sub=config.Z_SUB,
        c_adh=config.C_ADH,
        alpha=config.ALPHA_ADH
    )
    
    with torch.no_grad():
        # Compute Phase Difference (Ref - Loc)
        # [N, F]
        phase_loc = tri_layer_model_torch(f_tensor, k_tensor, **kwargs).detach()
        
        # Ref [1, F] -> broadcast
        # Make sure K_ref matches batch size or relies on broadcasting logic?
        # tri_layer_model_torch uses K_top.shape[0] as batch size. 
        # So we should match k_tensor shape for simplicity or just run once and broadcast later.
        k_ref_tensor = torch.full_like(k_tensor, 1e16)
        phase_ref = tri_layer_model_torch(f_tensor, k_ref_tensor, **kwargs).detach()
        
    phase_loc_np = phase_loc.cpu().numpy()
    phase_ref_np = phase_ref.cpu().numpy()

    # Unwrap individually
    loc_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_loc_np), axis=1))
    ref_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_ref_np), axis=1))
    
    # Difference (Ref - Loc)
    phase_diff = ref_uw - loc_uw
    
    # Center
    sim_phases_centered = phase_diff - np.mean(phase_diff, axis=1, keepdims=True)
    
    # Calculate stats for curves
    phase_mean = np.mean(sim_phases_centered, axis=0)
    phase_lower = np.percentile(sim_phases_centered, 2.5, axis=0)
    phase_upper = np.percentile(sim_phases_centered, 97.5, axis=0)

    if ax_phase is None:
        plt.figure(figsize=(10, 6))
        ax2 = plt.gca()
    else:
        ax2 = ax_phase

    ax2.plot(target_freqs/1e6, curve_centered, 'k-', linewidth=1.5, label='Exp')
    ax2.plot(target_freqs/1e6, phase_mean, 'r--', linewidth=1, label='Pred')
    ax2.fill_between(target_freqs/1e6, phase_lower, phase_upper, color='r', alpha=0.3)
    
    title_text = f'{filename}' if ax_phase else f"Reconstructed Phase with Uncertainty\nReference: {ref_name}"
    ax2.set_title(title_text, fontsize=10 if ax_phase else 12)
    ax2.set_xlabel("Freq (MHz)")
    if ax_phase is None:
        ax2.set_ylabel("Phase (Centered)")
        ax2.legend()
        
    ax2.grid(True, alpha=0.3)
    
    if ax_phase is None:
        phase_plot_path = f"results/uncertainty_phase_fit_{filename}.png"
        plt.savefig(phase_plot_path)
        print(f"Saved phase reconstruction to {phase_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, help="Path to Location CSV")
    parser.add_argument("--ref", type=str, help="Path to Reference CSV")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples for uncertainty analysis")
    args = parser.parse_args()
    
    uncertainty_task(args.loc, args.ref, args.samples)
