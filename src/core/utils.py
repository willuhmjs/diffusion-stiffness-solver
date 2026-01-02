import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import src.core.config as config
import os

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
            
        # Global Normalization
        # Using (x - mean) / global_std for uncentered data, but here data is centered??
        # WAIT: The generate.py subtracts global mean from raw data.
        # Here we subtracted local mean (line 53).
        # To match training distribution, we must use: (curve - global_mean) / global_std
        # BUT: The model is trained on mean-centered curves if we do curve - mean in generate.py?
        # Let's check generate.py again.
        # generate.py: X_final = (X_deg - phase_mean) / (phase_std + 1e-8)
        # It normalizes the RAW values globally.
        # So we should NOT subtract local mean here if we want to follow global norm exactly.
        # However, physics says phase offset is arbitrary.
        # If we remove local mean, we are effectively setting the DC component to 0.
        # If the training data includes DC offsets (it does, phase_mean is likely non-zero),
        # but the physics model returns absolute phase.
        # Experimental data has ARBITRARY offset.
        # So we MUST center the experimental curve (remove its arbitrary DC).
        # To match this, the training data should effectively be "centered" before global norm?
        # Or, we center here, then apply global scaling.
        # The key is AMPLITUDE scaling.
        # Correct approach for arbitrary offset data:
        # 1. Center the curve (remove DC).
        # 2. Scale by GLOBAL std dev (preserve relative amplitude).
        
        # So: curve_norm = curve_centered / (phase_std)
        # This assumes the training data was also roughly centered or that phase_std captures AC magnitude.
        
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
