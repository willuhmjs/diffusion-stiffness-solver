import torch
import src.core.config as config
import math

def get_cosine_schedule(timesteps):
    """
    Cosine schedule as proposed by Nichol & Dhariwal (2021).
    Better for small datasets than linear.
    """
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

# Pre-compute schedule
BETAS = get_cosine_schedule(config.TIMESTEPS)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, dim=0)

def add_noise(x_start, t, device=config.DEVICE):
    alphas_cumprod = ALPHAS_CUMPROD.to(device)
    
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t])
    
    noise = torch.randn_like(x_start)
    x_noisy = sqrt_alpha_bar.view(-1, 1) * x_start + sqrt_one_minus_alpha_bar.view(-1, 1) * noise
    return x_noisy, noise

def sample(model, condition_curve, num_samples=1, device=config.DEVICE):
    model.eval()
    betas = BETAS.to(device)
    alphas = ALPHAS.to(device)
    alphas_cumprod = ALPHAS_CUMPROD.to(device)
    
    with torch.no_grad():
        x = torch.randn(num_samples, 1).to(device)
        
        # Handle batching for condition curve
        if condition_curve.shape[0] != num_samples:
             condition_curve = condition_curve.repeat(num_samples, 1)
        
        for i in reversed(range(config.TIMESTEPS)):
            t = torch.tensor([i] * num_samples, device=device)
            t_norm = t.float().view(-1, 1) / config.TIMESTEPS
            
            predicted_noise = model(x, t_norm, condition_curve)
            
            alpha = alphas[i]
            alpha_bar = alphas_cumprod[i]
            beta = betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # Sampling calculation
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise
            
    return x
