import torch
from src.core.config_loader import cfg

# --- PHYSICS CONSTANTS ---
# Loaded from config/default.yaml
Z_SUB = float(cfg.physics.get('z_sub', 12.3e6))
C_SUB = float(cfg.physics.get('c_sub', 5640.0))
RHO_SUB = float(cfg.physics.get('rho_sub', 2180.0))

RHO_ADH = float(cfg.physics.get('rho_adh', 1290.0))
C_ADH = float(cfg.physics.get('c_adh', 1800.0))
Z_ADH = RHO_ADH * C_ADH 
ALPHA_ADH = float(cfg.physics.get('alpha_adh', 2500.0))

L_BL = float(cfg.physics.get('l_bl', 108.3e-6))

# --- DATA GENERATION ---
FREQ_MIN = float(cfg.physics.get('freq_min', 0.5e6))
FREQ_MAX = float(cfg.physics.get('freq_max', 15.0e6))
NUM_POINTS = int(cfg.physics.get('num_points', 2001))
DATASET_SIZE = int(cfg.data_generation.get('dataset_size', 5000))

K_MIN_LOG = float(cfg.physics.get('k_min_log', 12.0))
K_MAX_LOG = float(cfg.physics.get('k_max_log', 18.0))
K_MAX_PHYS = float(cfg.physics.get('k_max_phys', 1.0e16))

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = int(cfg.training.get('batch_size', 64))
LR = float(cfg.training.get('lr', 2e-4))
EPOCHS = int(cfg.training.get('epochs', 3000))
TIMESTEPS = int(cfg.training.get('timesteps', 500))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PATHS ---
DATA_PATH = cfg.paths.get('data_path', "data/processed/training_data.pt")
MODEL_PATH = cfg.paths.get('model_path', "checkpoints/model_final.pt")
