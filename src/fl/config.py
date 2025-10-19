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
        model_name = config["model_config"]["model_name"]
        num_rounds = config["fl"]["num_rounds"]
        num_clients = config["fl"]["num_clients"]
        per_round = config["fl"]["clients_per_round"]
        key = f"[{model_name}][rounds-{num_rounds}][epochs-{config['sft_config_args']['num_train_epochs']}][clients{num_clients}-per-round-{per_round}][{config['dataset']['labels_to_keep']}-{config['dataset']['classes_per_client']}][Lora-{config['use_lora']}]"
        if config['use_lora']:
            lora_r = config['lora_config']['r']
            lora_alpha = config['lora_config']['lora_alpha']
            key += f"[LoRA-r{lora_r}-alpha{lora_alpha}]"

        return sanitize_key(key)

    @staticmethod
    def load_config_with_corresponding_key(config_path=None):
        if config_path is None:
            config = ConfigManager.load_default_config()
            experiment_key = ConfigManager.generate_exp_key(config)

        else:
            raise NotImplementedError(
                "Custom config loading not yet implemented")

        ConfigManager.validate_config(config)
        return config, experiment_key


def get_default_config():
    return {
        "fl": {
            "num_rounds": 10,
            "num_clients": 25,
            "clients_per_round": 2
        },

        "sft_config_args": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 15,
            "num_train_epochs": 1,
            "learning_rate": 5e-5,
            "logging_steps": 20,
            "optim": "adamw_torch",
            "weight_decay": 0.001,
            "lr_scheduler_type": "constant",
            "seed": 42,
            "output_dir": None,
            "report_to": None,
            "disable_tqdm": False,
            "max_length": 512,
            "bf16": True,
            "save_strategy": "no",
            'dataset_num_proc': 4
            # 'assistant_only_loss': True
        },

        "model_config": {
            # 'google/gemma-3-270m-it',  "google/gemma-3-270m", "google/gemma-3-1b-pt",   "HuggingFaceTB/SmolLM3-3B-Base", "Qwen/Qwen3-0.6B-Base", "facebook/MobileLLM-R1-950M-base"
            "model_name": "google/gemma-3-270m-it",
        },

        "use_lora": False,

        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "bias": "none",
            # "finetune_vision_layers": False,
            # "finetune_language_layers": True,
            # "finetune_attention_modules": True,
            # "finetune_mlp_modules": True,
            "task_type": "CAUSAL_LM",
            'target_modules': [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            # "target_modules": 'all-linear'
        },


        "device": "cuda",
        "total_gpus": 1,
        "total_cpus": 10,
        "client_resources": {
            "num_cpus": 4,
            "num_gpus": 1
        },

        "dataset": {
            "samples_per_client": 512,
            "test_dataset_size": 512,
            "classes_per_client": 1,
            # "labels_to_keep": ['medical', 'finance'], # 93
            # "labels_to_keep": ['medical', 'math'], # 86
            # "labels_to_keep": ['finance', 'math'], # 80. accurate
            # 'labels_to_keep': ['chess', 'math'], # 77
            # 'labels_to_keep': ['math', 'coding'], #48
            "labels_to_keep": ['medical', 'finance', 'math']
            # "labels_to_keep": ['medical', 'finance']
        },
    }
