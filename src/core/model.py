import torch
import torch.nn as nn
import src.core.config as config

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, curve_points=config.NUM_POINTS, hidden_dim=256):
        super().__init__()
        
        # 1. ResNet-1D Feature Extractor
        # Much deeper and more robust than the previous simple CNN
        self.curve_encoder = nn.Sequential(
            # Input: [Batch, 1, 2001]
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3), # -> [32, 1001]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            ResidualBlock(32, 64, stride=2),   # -> [64, 501]
            ResidualBlock(64, 128, stride=2),  # -> [128, 251]
            ResidualBlock(128, 256, stride=2), # -> [256, 126]
            
            nn.AdaptiveAvgPool1d(1), # -> [256, 1] - Global Average Pooling
            nn.Flatten(),            # -> [256]
            nn.Linear(256, hidden_dim),
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
        # We increase depth here too for better reasoning
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
        # condition_curve: [Batch, Points]
        
        # Reshape curve for CNN: [Batch, 1, Points]
        if condition_curve.dim() == 2:
            condition_curve = condition_curve.unsqueeze(1)
            
        # Encode
        curve_emb = self.curve_encoder(condition_curve) # [Batch, Hidden]
        t_emb = self.time_mlp(t)                        # [Batch, Hidden]
        
        # Concatenate
        combined = torch.cat([x, curve_emb, t_emb], dim=1)
        
        return self.net(combined)
