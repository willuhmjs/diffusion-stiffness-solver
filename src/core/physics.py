import torch
import numpy as np
import src.core.config as config
from src.core.config_loader import cfg

def get_frequencies():
    """Generates the frequency tensor based on config."""
    # Create linear space of frequencies
    freqs = torch.linspace(config.FREQ_MIN, config.FREQ_MAX, config.NUM_POINTS)
    return freqs

def tri_layer_model_torch(
    frequencies,
    K_top,
    K_bottom=None,
    return_amp=False,
    alpha=None,
    c_adh=None,
):
    """
    Asymmetric Tri-Layer UT model using TMM.

    System:
    Substrate -> Interface (K_top) -> Adhesive -> Interface (K_bottom) -> Substrate

    If K_bottom is None, symmetric interface is assumed.
    """

    # -------------------------
    # Shape handling
    # -------------------------
    if K_top.dim() == 1:
        K_top = K_top.unsqueeze(1)

    if K_bottom is None:
        K_bottom = K_top
    elif K_bottom.dim() == 1:
        K_bottom = K_bottom.unsqueeze(1)

    w = 2 * np.pi * frequencies.unsqueeze(0)  # [1, F]
    j = torch.tensor(1j, dtype=torch.cfloat)

    batch_size = K_top.shape[0]
    num_freqs = frequencies.shape[0]

    Z_sub = float(config.Z_SUB)
    L_bl = float(config.L_BL)

    # -------------------------
    # Adhesive properties
    # -------------------------
    if c_adh is None:
        c_adh_val = float(config.C_ADH)
        Z_adh_val = float(config.Z_ADH)
    elif isinstance(c_adh, torch.Tensor):
        c_adh_val = c_adh
        if c_adh_val.dim() == 1:
            c_adh_val = c_adh_val.unsqueeze(1)
        rho_adh = float(config.RHO_ADH)
        Z_adh_val = rho_adh * c_adh_val
    else:
        c_adh_val = float(c_adh)
        rho_adh = float(config.RHO_ADH)
        Z_adh_val = rho_adh * c_adh_val

    if alpha is None:
        alpha_val = float(config.ALPHA_ADH)
    else:
        alpha_val = alpha

    # -------------------------
    # Spring matrices
    # -------------------------
    def spring_matrix(K):
        M = torch.zeros(batch_size, num_freqs, 2, 2, dtype=torch.cfloat)
        M[:, :, 0, 0] = 1.0
        M[:, :, 1, 1] = 1.0
        M[:, :, 1, 0] = j * w / K
        return M

    M_K_top = spring_matrix(K_top)
    M_K_bot = spring_matrix(K_bottom)

    # -------------------------
    # Adhesive layer matrix
    # -------------------------
    k_adh = (w / c_adh_val) - j * alpha_val
    arg = k_adh * L_bl

    c_term = torch.cos(arg)
    s_term = torch.sin(arg)

    if c_term.dim() == 1:
        c_term = c_term.unsqueeze(0).expand(batch_size, -1)
        s_term = s_term.unsqueeze(0).expand(batch_size, -1)

    M_layer = torch.zeros(batch_size, num_freqs, 2, 2, dtype=torch.cfloat)
    M_layer[:, :, 0, 0] = c_term
    M_layer[:, :, 1, 1] = c_term

    if isinstance(Z_adh_val, torch.Tensor):
        # Ensure broadcasting capability [B, F]
        if Z_adh_val.dim() == 1:
             Z_adh_val = Z_adh_val.unsqueeze(1)
        Z_exp = Z_adh_val.expand(-1, num_freqs)
        M_layer[:, :, 0, 1] = j * Z_exp * s_term
        M_layer[:, :, 1, 0] = j * s_term / Z_exp
    else:
        M_layer[:, :, 0, 1] = j * Z_adh_val * s_term
        M_layer[:, :, 1, 0] = j * s_term / Z_adh_val

    # -------------------------
    # Total transfer matrix
    # -------------------------
    M_total = torch.matmul(torch.matmul(M_K_top, M_layer), M_K_bot)

    A = M_total[:, :, 0, 0]
    B = M_total[:, :, 0, 1]
    C = M_total[:, :, 1, 0]
    D = M_total[:, :, 1, 1]

    Z_in = (A * Z_sub + B) / (C * Z_sub + D)
    R = (Z_in - Z_sub) / (Z_in + Z_sub)

    phase_deg = torch.rad2deg(torch.angle(R))

    if return_amp:
        return phase_deg, torch.abs(R)

    return phase_deg

def add_noise(phase_curves, frequencies):
    """
    Injects realistic sensor noise: Gaussian Noise + Baseline Drift
    """
    noise_cfg = cfg.data_generation.get('noise', {})
    if not noise_cfg.get('enable', False):
        return phase_curves

    batch_size, num_points = phase_curves.shape
    
    # 1. Gaussian Noise (High Frequency)
    sigma = noise_cfg.get('sigma_phase', 0.5)
    gaussian_noise = torch.randn_like(phase_curves) * sigma
    
    # 2. Baseline Drift (Low Frequency)
    # Simulates coupling wobble using a Sine wave with random phase/period
    drift_mag = noise_cfg.get('drift_factor', 1.0)
    
    # Random drift period between 1/2 and 2x the bandwidth
    t = torch.linspace(0, 1, num_points).unsqueeze(0).expand(batch_size, -1)
    drift_freq = torch.rand(batch_size, 1) * 2.0 + 0.5 # 0.5 to 2.5 cycles
    drift_phase = torch.rand(batch_size, 1) * 2 * np.pi
    
    drift = drift_mag * torch.sin(2 * np.pi * drift_freq * t + drift_phase)
    
    return phase_curves + gaussian_noise + drift

def generate_dataset(n_samples=1000, asymmetric=True):
    print(f"Generating {n_samples} physics samples...")

    log_k = torch.FloatTensor(n_samples, 1).uniform_(
        config.K_MIN_LOG, config.K_MAX_LOG
    )
    K_top = 10 ** log_k

    if asymmetric:
        # Bottom interface varies independently within ±1 decade
        delta = torch.FloatTensor(n_samples, 1).uniform_(-1.0, 1.0)
        K_bottom = K_top * (10 ** delta)
    else:
        K_bottom = K_top

    # Disbonds (15%)
    n_disbonds = int(0.15 * n_samples)
    if n_disbonds > 0:
        log_k_weak = torch.FloatTensor(n_disbonds, 1).uniform_(6.0, 11.0)
        K_top[:n_disbonds] = 10 ** log_k_weak
        K_bottom[:n_disbonds] = 10 ** log_k_weak

    alpha = torch.FloatTensor(n_samples, 1).uniform_(
        config.ALPHA_ADH * 0.5, config.ALPHA_ADH * 2.0
    )

    c_adh = torch.FloatTensor(n_samples, 1).uniform_(
        config.C_ADH * 0.9, config.C_ADH * 1.1
    )

    freqs = get_frequencies()

    phase = tri_layer_model_torch(
        freqs,
        K_top,
        K_bottom=K_bottom,
        alpha=alpha,
        c_adh=c_adh,
    )

    phase_noisy = add_noise(phase, freqs)

    return phase_noisy, K_top, K_bottom, freqs
