import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.core.utils as utils
from src.core.model import ConditionalDiffusionModel
from src.core.diffusion import sample
from src.core.physics import tri_layer_model_torch
from src.core.config_loader import cfg

def load_config_data(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_model_and_stats(config_data, device):
    # Load Stats
    data_path = config_data['paths']['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return None, None
        
    try:
        data = torch.load(data_path, map_location=device, weights_only=True)
        stats = data['stats']
    except Exception as e:
        print(f"Error loading stats from {data_path}: {e}")
        return None, None

    # Load Model
    model_path = config_data['paths']['model_path']
    model = ConditionalDiffusionModel().to(device)
    
    if not os.path.exists(model_path):
        # Fallback logic based on naming convention if exact path missing
        if 'set2' in model_path:
            fallback = 'checkpoints/model_set2.pt'
        else:
            fallback = 'checkpoints/model_set1.pt'
            
        if os.path.exists(fallback):
            print(f"Model path {model_path} not found. Using fallback: {fallback}")
            model_path = fallback
        else:
            print(f"Error: Model path {model_path} and fallback not found.")
            return None, None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded Model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
        
    return model, stats

def simulate_phase(k_val, freqs, physics_params, l_bl_override=None):
    """Simulate phase difference (ref - loc) matching experimental sign convention."""
    k_tensor = torch.tensor([k_val]).float()
    
    # Extract params
    l_bl = l_bl_override if l_bl_override is not None else physics_params.get('l_bl')
    c_adh = physics_params.get('c_adh')
    alpha_adh = physics_params.get('alpha_adh')
    z_sub = physics_params.get('z_sub')

    common_kwargs = dict(
        alpha=float(alpha_adh) if alpha_adh else None,
        c_adh=float(c_adh) if c_adh else None,
        l_bl=float(l_bl) if l_bl else None,
        z_sub=float(z_sub) if z_sub else None,
    )
    
    # Phase at location (predicted K)
    phase_loc = tri_layer_model_torch(
        freqs, k_tensor, **common_kwargs
    )
    
    # Phase at reference (perfect bond)
    k_ref = torch.tensor([1e16]).float()
    phase_ref = tri_layer_model_torch(
        freqs, k_ref, **common_kwargs
    )
    
    # Phase difference: ref - loc (matches experimental loc - ref sign convention)
    phase_diff = phase_ref - phase_loc
    phase_np = phase_diff.detach().cpu().numpy().flatten()
    
    # Processing: deg -> rad -> unwrap -> deg -> center
    phase_rad = np.deg2rad(phase_np)
    phase_unwrapped = np.unwrap(phase_rad)
    phase_deg_final = np.rad2deg(phase_unwrapped)
    phase_centered = phase_deg_final - np.mean(phase_deg_final)
    
    return phase_centered

def run_batch_inference():
    print("\n--- BATCH INFERENCE (SINGLE CONFIG) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Configs
    config_data = cfg._config
    
    # 2. Load Model
    print("\nLoading Model...")
    model, stats = load_model_and_stats(config_data, device)
    
    if model is None:
        print("Error: Could not load model. Aborting.")
        return

    # 3. Identify Files
    data_dir = 'data/raw'
    tasks = utils.find_specimen_pairs(data_dir)
    print(f"\nFound {len(tasks)} file pairs to process.")

    # Load Specimen Properties
    thickness_map = {}
    try:
        meta_path = 'data/metadata/specimen_properties.csv'
        if os.path.exists(meta_path):
            df_props = pd.read_csv(meta_path)
            # Create a lookup dictionary: (Specimen, Location) -> Thickness
            # Check if column is 'Thickness_m' or 'Thickness, m'
            thick_col = 'Thickness_m' if 'Thickness_m' in df_props.columns else 'Thickness, m'
            thickness_map = df_props.set_index(['Specimen', 'Location'])[thick_col].to_dict()
            print("Loaded specimen properties.")
        else:
            print(f"Warning: Metadata file not found at {meta_path}")
    except Exception as e:
        print(f"Warning: Could not load specimen properties: {e}")
    
    results = []
    
    # 4. Processing Loop
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Processing {task['specimen']} Location {task['location']} (File: {task['filename']})...")
        
        try:
            # Determine Thickness Information
            spec_key = task['specimen']
            loc_key = task['location']
            s_num = None
            l_num = None
            
             # Normalize keys for lookup
            try:
                if isinstance(spec_key, str) and spec_key.lower().startswith("spec"):
                     s_num = int(spec_key[4:])
                else:
                     s_num = int(spec_key)
                l_num = int(loc_key)
                
                h_loc = thickness_map.get((s_num, l_num))
            except:
                h_loc = None
            
            # Load Data
            df_loc = utils.load_file_to_dataframe(task['loc_path'])
            df_ref = utils.load_file_to_dataframe(task['ref_path'])
            df_loc.columns = df_loc.columns.str.strip()
            df_ref.columns = df_ref.columns.str.strip()
            
            real_freqs = df_loc['Frequency'].values * 1e6
            
            # --- SIGN FLIP DIAGNOSIS ---
            # Using (Ref - Loc) to flip the sign for experiment:
            raw_phase_diff = df_ref['Phase'].values - df_loc['Phase'].values
            
            # Process for Model
            curve_tensor, curve_centered, target_freqs = utils.process_experimental_data(
                real_freqs, raw_phase_diff, stats=stats
            )
            
            freqs_tensor = torch.tensor(target_freqs).float() # For physics sim
            
            # Check Quality
            ptp = curve_centered.max() - curve_centered.min()
            if ptp < 0.5:
                print(f"  -> Skipped: Low amplitude ({ptp:.2f} deg)")
                continue

            # --- RUN MODEL ---
            cond = curve_tensor.to(device).unsqueeze(0)
            
            # inference.py samples 50 times.
            preds = sample(model, cond, num_samples=50, device=device)
            k_vals = [utils.inverse_transform_k(p, stats) for p in preds]
            
            # Apply clamping
            import src.core.config as config_module
            K_MAX = config_module.K_MAX_PHYS
            
            k_vals_clamped = [min(k, K_MAX) for k in k_vals]
            mean_k = float(np.mean(k_vals_clamped))
            std_k = float(np.std(k_vals_clamped))
            
            real_thickness = h_loc # Reuse determined thickness for sim

            # Simulate for verification
            # Pass l_bl_override as real_thickness
            sim = simulate_phase(mean_k, freqs_tensor, config_data['physics'], l_bl_override=real_thickness)
            
            # Compare directly without amplitude scaling (both are centered phase diff in degrees)
            mse = np.mean((curve_centered - sim)**2)
            
            print(f"  -> Model (K={mean_k:.2e}, MSE={mse:.4f}, Th={real_thickness if real_thickness else 'Def'})")
            
            results.append({
                'Specimen': task['specimen'],
                'Location': task['location'],
                'Predicted_K': mean_k,
                'Uncertainty': std_k, 
                'Config_Used': 'Default',
                'MSE': mse,
                'Thickness_Used': real_thickness if real_thickness else 'Default'
            })
            
            # --- PLOTTING ---
            os.makedirs('results/fits', exist_ok=True)
            fit_path = os.path.join('results/fits', f"fit_{task['specimen']}_Loc{task['location']}.png")
            
            # Scaling for visual verification (like in inference.py)
            scaling_factor = (np.max(curve_centered) - np.min(curve_centered)) / \
                             (np.max(sim) - np.min(sim)) if (np.max(sim) - np.min(sim)) > 1e-6 else 1.0
            
            sim_scaled = sim * scaling_factor

            plt.figure(figsize=(10,6))
            plt.plot(target_freqs/1e6, curve_centered, 'b-', label='Real Data')
            # Plot Scaled AI Pred like inference.py
            plt.plot(target_freqs/1e6, sim_scaled, 'r--', label=f'AI Pred (K={mean_k:.1e}) [Scaled]')
            
            plt.title(f"Model Fit\nSpecimen {task['specimen']} Loc {task['location']}\nThickness: {real_thickness if real_thickness else 'Default'}")
            plt.xlabel("Freq (MHz)")
            plt.ylabel("Phase Deviation")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(fit_path)
            plt.close()
            
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Results
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv('results/batch_inference_results.csv', index=False)
        
        # Summary
        summary = df_res.groupby('Specimen')['Predicted_K'].median().reset_index()
        summary.rename(columns={'Predicted_K': 'Median_K'}, inplace=True)
        summary.to_csv('results/batch_inference_summary.csv', index=False)
        print("\n--- Summary ---")
        print(summary)

if __name__ == "__main__":
    run_batch_inference()
