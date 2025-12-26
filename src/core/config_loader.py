import os
import yaml
import torch

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        default_path = os.path.join(os.path.dirname(__file__), "../../config/default.yaml")
        config_path = os.getenv('CONFIG_PATH', default_path)
        
        # Ensure absolute path if relative path provided via env var
        if not os.path.isabs(config_path) and config_path != default_path:
             config_path = os.path.abspath(config_path)

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            print(f"Loaded config from: {config_path}")
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            self._config = {} # Fallback or Raise Error

    def get(self, section, key, default=None):
        return self._config.get(section, {}).get(key, default)

    @property
    def physics(self):
        return self._config.get('physics', {})

    @property
    def training(self):
        return self._config.get('training', {})
    
    @property
    def data_generation(self):
        return self._config.get('data_generation', {})
    
    @property
    def paths(self):
        return self._config.get('paths', {})

# Singleton Accessor
cfg = ConfigLoader()
