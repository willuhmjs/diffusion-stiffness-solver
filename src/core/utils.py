import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import src.core.config as config
import os
import re

def load_file_to_dataframe(path):
    """
    Loads a file (CSV or DAT) into a pandas DataFrame.
    Handles .dat files with whitespace separators and specific column renaming.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.dat':
        try:
            # Read .dat file with whitespace separator
            df = pd.read_csv(path, sep=r'\s+', engine='python')
            
            # Rename columns to match internal standard
            # Mapping based on typical .dat file structure from import_data.py
            rename_map = {
                'Freq(MHz)': 'Frequency',
                'Frequency': 'Frequency', # Sometimes it is already Frequency
                'Mag': 'Amp',
                'Phs(rad)': 'Phase',
                'Phs': 'Phase'
            }
            
            df.rename(columns=rename_map, inplace=True)
            
            # Ensure required columns exist
            required = ['Frequency', 'Amp', 'Phase']
            if not all(col in df.columns for col in required):
                # Try simple position-based if names fail?
                # For now, just warn/fail if names are wrong
                pass
                
            return df
        except Exception as e:
            print(f"Error reading .dat file {path}: {e}")
            raise e
            
    else:
        # Default to CSV
        return pd.read_csv(path)

def find_specimen_pairs(data_dir):
    """
    Scans the directory for Spec*_Loc*_Rep*.(csv|dat) files and their corresponding Ref files.
    Returns a list of tuples: (loc_path, ref_path, info_dict)
    """
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return []

    all_files = os.listdir(data_dir)
    # Match both .csv and .dat
    pattern = re.compile(r'(Spec\d+)_Loc(\d+)_Rep(\d+)\.(csv|dat)', re.IGNORECASE)
    found_pairs = []
    
    for f in all_files:
        match = pattern.match(f)
        if match:
            specimen = match.group(1)
            location = match.group(2)
            rep = match.group(3)
            ext = match.group(4) # csv or dat
            
            loc_path = os.path.join(data_dir, f)
            
            # Look for Ref file with same extension first
            ref_filename = f"{specimen}_Ref_Rep{rep}.{ext}"
            ref_path = os.path.join(data_dir, ref_filename)
            
            # If not found, try the other extension (though usually they match)
            if not os.path.exists(ref_path):
                other_ext = 'dat' if ext.lower() == 'csv' else 'csv'
                ref_filename_alt = f"{specimen}_Ref_Rep{rep}.{other_ext}"
                ref_path_alt = os.path.join(data_dir, ref_filename_alt)
                if os.path.exists(ref_path_alt):
                    ref_path = ref_path_alt
            
            # Fallback: If no reference file specific to this specimen/rep exists,
            # try to use a common reference or a reference from another specimen if appropriate.
            # For this dataset, Spec4 often reuses Spec3's reference or a specific one.
            # Let's check for Spec3_Ref_Rep1.dat as a fallback if Spec4 is missing one.
            
            if not os.path.exists(ref_path):
                 if specimen == 'Spec4':
                     # Try using Spec3's reference for Spec4 if Spec4's own ref is missing
                     fallback_ref = os.path.join(data_dir, f"Spec3_Ref_Rep{rep}.{ext}")
                     if os.path.exists(fallback_ref):
                         ref_path = fallback_ref
                         # print(f"Info: Using fallback reference {fallback_ref} for {f}")

            if os.path.exists(ref_path):
                found_pairs.append({
                    'loc_path': loc_path,
                    'ref_path': ref_path,
                    'specimen': specimen,
                    'location': location,
                    'rep': rep,
                    'filename': f
                })

    
    # Sort for consistency
    found_pairs.sort(key=lambda x: (x['specimen'], int(x['location']), int(x['rep'])))
    return found_pairs

def load_stats(device=config.DEVICE):
    """
    Loads the normalization statistics from the processed data file.
    """
    if not os.path.exists(config.DATA_PATH):
        print(f"Error: {config.DATA_PATH} not found. Run generate_data.py first.")
        return None
        
    try:
        data = torch.load(config.DATA_PATH, map_location=device, weights_only=True)
        return data['stats']
    except Exception as e:
        print(f"Error loading stats: {e}")
        return None

def process_experimental_data(freqs, phase_diff, stats=None, target_freqs=None):
    """
    Interpolates, Centers, and Normalizes experimental data.
    
    Args:
        freqs (array): Frequencies of the input data.
        phase_diff (array): Phase difference values.
        stats (dict, optional): Normalization statistics (mean/std). 
                              If None, uses Instance Normalization (Not Recommended for Amplitude tasks).
        target_freqs (array, optional): Frequencies to interpolate to. Defaults to config range.
        
    Returns:
        tuple: (normalized_tensor, centered_curve, target_freqs)
    """
    if target_freqs is None:
        target_freqs = np.linspace(config.FREQ_MIN, config.FREQ_MAX, config.NUM_POINTS)
        
    # Unwrapping: Fix phase jumps > pi
    phase_diff = np.unwrap(phase_diff)

    # Smoothing: Apply Gaussian Filter to reduce noise
    # Standard deviation of 2.0 corresponds to a mild smoothing to suppress high-freq jitter
    phase_diff_smooth = gaussian_filter1d(phase_diff, sigma=2.0)

    # Interpolate
    f_interp = interp1d(freqs, phase_diff_smooth, kind='linear', fill_value="extrapolate")
    curve = f_interp(target_freqs)
    
    # Center (Remove Mean) - This removes the arbitrary phase offset
    # Note: Global Normalization typically happens on Centered Data in this pipeline
    curve_centered = curve - np.mean(curve)
    
    # Normalize
    if stats is not None and 'phase_mean' in stats and 'phase_std' in stats:
        # Extract scalar values from stats (which might be 0-dim tensors or python floats)
        phase_mean = stats['phase_mean']
        phase_std = stats['phase_std']
        
        if isinstance(phase_mean, torch.Tensor): phase_mean = phase_mean.item()
        if isinstance(phase_std, torch.Tensor): phase_std = phase_std.item()
            
        
        curve_norm = curve_centered / (phase_std + 1e-8)
        
    else:
        # Instance Normalize (Fallback) - Destroys Amplitude Info!
        print("Warning: No Global Stats provided. Using Instance Normalization (may reduce accuracy).")
        curve_std = np.std(curve_centered)
        curve_norm = curve_centered / (curve_std + 1e-8)
    
    # Convert to Tensor
    curve_tensor = torch.tensor(curve_norm, dtype=torch.float32)
    
    return curve_tensor, curve_centered, target_freqs

def inverse_transform_k(log_k_norm, stats):
    """
    Converts model output (normalized log k) back to real K.
    """
    if isinstance(log_k_norm, torch.Tensor):
        log_k_norm = log_k_norm.item()
        
    # Un-normalize: x * std + mean
    k_std = stats['k_std']
    k_mean = stats['k_mean']

    if isinstance(k_std, torch.Tensor): k_std = k_std.item()
    if isinstance(k_mean, torch.Tensor): k_mean = k_mean.item()

    log_real = (log_k_norm * k_std) + k_mean
    return 10 ** log_real
