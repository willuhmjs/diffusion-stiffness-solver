import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    z_sub = physics_params.get('z_sub')
    l_bl = l_bl_override if l_bl_override is not None else physics_params.get('l_bl')
    c_adh = physics_params.get('c_adh')
    alpha = physics_params.get('alpha_adh')
    
    common_kwargs = dict(
        alpha=float(alpha) if alpha else None,
        c_adh=float(c_adh) if c_adh else None,
        z_sub=float(z_sub) if z_sub else None,
        l_bl=float(l_bl) if l_bl else None,
    )
    
    # Phase at location (predicted K)
    phase_loc = tri_layer_model_torch(
        freqs, K_top=k_tensor, K_bottom=k_tensor, **common_kwargs
    )
    
    # Phase at reference (perfect bond)
    k_ref = torch.tensor([1e16]).float()
    phase_ref = tri_layer_model_torch(
        freqs, K_top=k_ref, K_bottom=k_ref, **common_kwargs
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
    median_thickness_map = {}
    try:
        df_props = pd.read_csv('data/metadata/specimen_properties.csv')
        # Create a lookup dictionary: (Specimen, Location) -> Thickness
        thickness_map = df_props.set_index(['Specimen', 'Location'])['Thickness, m'].to_dict()
        
        # Calculate Median Thickness per Specimen for Reference Estimation
        median_thickness_map = df_props.groupby('Specimen')['Thickness, m'].median().to_dict()
        print("Loaded specimen properties.")
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
            
             # Normalize keys for lookup
            try:
                if isinstance(spec_key, str) and spec_key.lower().startswith("spec"):
                     s_num = int(spec_key[4:])
                else:
                     s_num = int(spec_key)
                l_num = int(loc_key)
                
                h_loc = thickness_map.get((s_num, l_num))
                h_ref = median_thickness_map.get(s_num)
            except:
                h_loc = None
                h_ref = None

            # Load Data
            df_loc = utils.load_file_to_dataframe(task['loc_path'])
            df_ref = utils.load_file_to_dataframe(task['ref_path'])
            df_loc.columns = df_loc.columns.str.strip()
            df_ref.columns = df_ref.columns.str.strip()
            
            real_freqs = df_loc['Frequency'].values * 1e6
            
            # --- SIGN FLIP DIAGNOSIS ---
            # Previous logic assumed Exp(Loc-Ref) == TMM(Ref-Loc) due to sign convention.
            # However, results showed inverted correlation (Strong Spec1 -> Weak Pred).
            # We explicitly flip the sign here to test if Exp(Ref-Loc) aligns better.
            # Using (Ref - Loc) from experiment:
            raw_phase_diff = df_ref['Phase'].values - df_loc['Phase'].values
            
            # --- PHYSICS CORRECTION (Thickness Mismatch) ---
            if h_loc and h_ref:
            # if False: # TEMPORARILY DISABLED TO CHECK BASELINE
                # Calculate what the Reference Phase would be if it had h_loc instead of h_ref
                # Correction = Phase(Ref, h_loc) - Phase(Ref, h_ref)
                
                # Use typical params from config
                # We need frequencies that match the REAL data for correction, or interp later?
                # Usually simpler to generate correction on target_freqs and interp raw_phase_diff, 
                # OR generate correction on real_freqs. 
                # Physics model is fast, let's generate on target_freqs (which process_experimental_data uses).
                
                # Wait, utils.process_experimental_data handles interpolation. 
                # Better to correct AFTER processing/interpolation so freqs match.
                pass
            
            # Process for Model
            curve_tensor, curve_centered, target_freqs = utils.process_experimental_data(
                real_freqs, raw_phase_diff, stats=stats
            )
            
            # --- APPLY CORRECTION ---
            if h_loc and h_ref:
                # print(f"    Applying Thickness Correction: h_loc={h_loc*1e6:.1f}um, h_ref~={h_ref*1e6:.1f}um")
                
                freqs_t = torch.tensor(target_freqs).float()
                k_good = torch.tensor([1e16]).float()
                
                # Get common physics params
                phys = config_data.get('physics', {})
                params = {
                    'c_adh': float(phys.get('c_adh', 2650.0)),
                    'alpha': float(phys.get('alpha_adh', 1000.0)),
                    'z_sub': float(phys.get('z_sub', 1.76e7))
                }
                
                # Phase(Ref, h_loc)
                p_ref_hloc = tri_layer_model_torch(freqs_t, k_good, k_good, l_bl=float(h_loc), **params)
                # Phase(Ref, h_ref)
                p_ref_href = tri_layer_model_torch(freqs_t, k_good, k_good, l_bl=float(h_ref), **params)
                
                correction = (p_ref_hloc - p_ref_href).detach().numpy().flatten()
                
                # Unwrap correction (it might wrap if h diff is large, though unlikely for 20um)
                correction = np.rad2deg(np.unwrap(np.deg2rad(correction)))
                
                # Apply Correction: Target = Measured + Correction
                # curve_centered is in degrees
                curve_corrected = curve_centered + correction
                
                # Re-center (model requires zero mean)
                curve_corrected_centered = curve_corrected - np.mean(curve_corrected)
                
                # Update input tensor
                # Normalize using stats
                p_mean = stats['phase_mean']
                p_std = stats['phase_std']
                if isinstance(p_mean, torch.Tensor): p_mean = p_mean.item()
                if isinstance(p_std, torch.Tensor): p_std = p_std.item()
                
                curve_norm = (curve_corrected_centered - p_mean) / (p_std + 1e-8)
                curve_tensor = torch.tensor(curve_norm, dtype=torch.float32)
                curve_centered = curve_corrected_centered # For plotting/MSE
            
            freqs_tensor = torch.tensor(target_freqs).float() # For physics sim
            
            # Check Quality
            ptp = curve_centered.max() - curve_centered.min()
            if ptp < 0.5:
                print(f"  -> Skipped: Low amplitude ({ptp:.2f} deg)")
                continue

            # --- RUN MODEL ---
            cond = curve_tensor.to(device).unsqueeze(0)
            preds = sample(model, cond, num_samples=50, device=device)
            k_vals = [utils.inverse_transform_k(p, stats) for p in preds]
            # Use median to reject outlier samples (e.g. degenerate high-K predictions)
            mean_k = float(np.median(k_vals))
            
            real_thickness = h_loc # Reuse determined thickness for sim

            if real_thickness:
                # print(f"    Using measured thickness: {real_thickness:.6f} m")
                pass
            else:
                pass


            # Simulate
            sim = simulate_phase(mean_k, freqs_tensor, config_data['physics'], l_bl_override=real_thickness)
            # Compare directly without amplitude scaling (both are centered phase diff in degrees)
            mse = np.mean((curve_centered - sim)**2)
            
            print(f"  -> Model (K={mean_k:.2e}, MSE={mse:.4f}, Th={real_thickness if real_thickness else 'Def'})")
            
            results.append({
                'Specimen': task['specimen'],
                'Location': task['location'],
                'Predicted_K': mean_k,
                'Config_Used': 'Default',
                'MSE': mse,
                'Thickness_Used': real_thickness if real_thickness else 'Default'
            })
            
            # --- PLOTTING ---
            os.makedirs('results/fits', exist_ok=True)
            fit_path = os.path.join('results/fits', f"fit_{task['specimen']}_Loc{task['location']}.png")
            
            plt.figure(figsize=(10,6))
            plt.plot(target_freqs/1e6, curve_centered, 'b-', label='Real Data')
            plt.plot(target_freqs/1e6, sim, 'r--', label=f'AI Pred (K={mean_k:.1e})')
            
            # Reference Line (from Fracture Energy) removed
            
            plt.title(f"Model Fit\nSpecimen {task['specimen']} Loc {task['location']}")
            plt.xlabel("Freq (MHz)")
            plt.ylabel("Phase Deviation")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(fit_path)
            plt.close()
            
        except Exception as e:
            print(f"  -> Error: {e}")

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
