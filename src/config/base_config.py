from src.config.default import get_default_config
from src.utils.utils import sanitize_key

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
    def generate_exp_key(config):
        model_name = config["model_config"]["model_name"]  # e.g., "gemma-3-270m-it"
        num_rounds = config["fl"]["num_rounds"]
        num_clients = config["fl"]["num_clients"]
        client_0_dataset = config["dataset"]["client_0_dataset"]
        client_1_dataset = config["dataset"]["client_1_dataset"]
        key = f"[{model_name}][rounds{num_rounds}][epochs-{config['sft_config_args']['num_train_epochs']}][clients{num_clients}][C0-{client_0_dataset}-C1{client_1_dataset}]"
        if config['use_lora']:
            lora_r = config['lora_config']['r']
            lora_alpha = config['lora_config']['lora_alpha']
            key += f"[LoRA-r{lora_r}-alpha{lora_alpha}][New2]"
        
        return sanitize_key(key)
    
    @staticmethod
    def load_config_with_corresponding_key(config_path=None):
        if config_path is None:
            config = ConfigManager.load_default_config()
            experiment_key = ConfigManager.generate_exp_key(config)

        else:
            raise NotImplementedError("Custom config loading not yet implemented")
        
        ConfigManager.validate_config(config)
        return config, experiment_key
