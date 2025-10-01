from src.config.default import get_default_config

class ConfigManager:
    @staticmethod
    def load_default_config():
        return get_default_config()
    
    @staticmethod
    def validate_config(config):
        required_keys = ["fl", "sft_config_args", "model_config", "device"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    @staticmethod
    def load_config(config_path=None):
        if config_path is None:
            config = ConfigManager.load_default_config()
        else:
            raise NotImplementedError("Custom config loading not yet implemented")
        
        ConfigManager.validate_config(config)
        return config
