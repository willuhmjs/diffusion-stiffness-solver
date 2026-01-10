import torch
import torch.nn as nn
import src.core.config as config


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, curve_points=config.NUM_POINTS, hidden_dim=256):
        super().__init__()
        
        # 1. Direct Linear Projection for Physics Data
        # Pass the raw phase curve directly to the diffusion model
        # Input: [Batch, curve_points] -> Output: [Batch, hidden_dim]
        self.curve_encoder = nn.Sequential(
            nn.Linear(curve_points, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 3. Denoising Head
        # Concatenates: [Noisy_K (1) + Curve_Features (hidden) + Time_Info (hidden)]
        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) # Output: Predicted Noise
        )

    def forward(self, x, t, condition_curve):
        # condition_curve: [Batch, Points] or [Batch, 1, Points]
        
        # Handle case where curve was reshaped for CNN (legacy compatibility)
        if condition_curve.dim() == 3:
            condition_curve = condition_curve.squeeze(1)
            
        # Encode the raw curve directly
        curve_emb = self.curve_encoder(condition_curve) # [Batch, Hidden]
        t_emb = self.time_mlp(t)                        # [Batch, Hidden]
        
        # Concatenate
        combined = torch.cat([x, curve_emb, t_emb], dim=1)
        
        return self.net(combined)
