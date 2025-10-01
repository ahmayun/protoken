def get_default_config():
    return {
        "fl": {
            "num_rounds": 17,
            "num_clients": 2,
            "clients_per_round": 2
        },

        "sft_config_args": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 15,
            "num_train_epochs": 1,
            "learning_rate": 1e-5,
            "logging_steps": 20,
            "optim": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "constant",
            "seed": 3407,
            "output_dir": None,
            "report_to": None,
        },

        "model_config": {
            "model_name": "google/gemma-3-270m", # HuggingFaceTB/SmolLM3-3B-Base, Qwen/Qwen3-0.6B-Base, facebook/MobileLLM-R1-950M-base
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "full_finetuning": True,
        },

        "chat_template": "gemma3",

        "device": "cuda",
        "total_gpus": 1,
        "total_cpus": 10,
        "client_resources": {
            "num_cpus": 4,
            "num_gpus": 0.5
        },

        "dataset":{
            "client_dataset_size": 2048,
            "test_dataset_size": 512
        }
    }
