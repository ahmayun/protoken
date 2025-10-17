def get_default_config():
    return {
        "fl": {
            "num_rounds": 4,
            "num_clients": 4,
            "clients_per_round": 4
        },

        "sft_config_args": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 15,
            "num_train_epochs": 1,
            "learning_rate": 5e-5,
            "logging_steps": 20,
            "optim": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "constant",
            "seed": 3407,
            "output_dir": None,
            "report_to": None,
            "disable_tqdm": False,
            "max_length": 512,
            "bf16": True,
            "save_strategy": "no", 
            'dataset_num_proc':4
            # 'assistant_only_loss': True
        },

        "model_config": {
            #'google/gemma-3-270m-it',  "google/gemma-3-270m", "google/gemma-3-1b-pt",   "HuggingFaceTB/SmolLM3-3B-Base", "Qwen/Qwen3-0.6B-Base", "facebook/MobileLLM-R1-950M-base"
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
            "client_dataset_size": 2048,
            "test_dataset_size": 512,
            "classes_per_client": 1
        },
    }
