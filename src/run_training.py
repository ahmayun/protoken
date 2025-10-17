import time
import torch
import gc
from src.fl.simulation import run_fl_experiment
from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.config.config import ConfigManager

def _get_experiment_matrix():
    experiments = []
    MODELS = [
        "google/gemma-3-270m-it",
        # "google/gemma-3-1b-it",
        # "Qwen/Qwen3-0.6B"
    ]

    for model in MODELS:
        experiment = {
            "model_name": model,
        }
        experiments.append(experiment)

    return experiments

def _generate_experiment_config(experiment_setting_dict, use_lora, epochs):
    config = ConfigManager.load_default_config()
    config["model_config"]["model_name"] = experiment_setting_dict["model_name"]
    config["use_lora"] = use_lora
    config['sft_config_args']['num_train_epochs'] = epochs
    return config

def single_exp_run(config):
    experiment_key = ConfigManager.generate_exp_key(config)

    print(f"Running: {experiment_key}")

    # if CacheManager.experiment_is_complete(experiment_key):
    #     print("✅ Experiment already completed. Skipping...")
    #     return

    start_time = time.time()
    metrics = run_fl_experiment(config)
    duration = time.time() - start_time

    CacheManager.consolidate_experiment(experiment_key, config, metrics)
    save_json({"metrics": metrics, 'config': config} , f"results/fl_train_metrics_{experiment_key}.json")
    print(f"✅ Completed in {duration:.1f}s")
    del metrics
    torch.cuda.empty_cache()
    gc.collect()
    

def run_experiments():
    all_experiments = _get_experiment_matrix()
    epochs = 1
    for i, exp in enumerate(all_experiments):
        print(f"{i} Experiment Key: {ConfigManager.generate_exp_key(_generate_experiment_config(exp, use_lora=True, epochs=epochs))}")
    
    _ = input("Only for test. Press enter to continue.")

    for i, exp in enumerate(all_experiments):
        config = _generate_experiment_config(exp, use_lora=True, epochs=epochs)
        print(f"Experiment [{i}/{len(all_experiments)}]")
        single_exp_run(config)
    
    print(f"\n🎉 All {len(all_experiments)} experiments completed!")
    

if __name__ == "__main__":
    run_experiments()
