import time
import torch
import gc
from src.fl.simulation import run_fl_experiment
from src.utils.utils import CacheManager, save_json
from src.config.base_config import ConfigManager

import os
import multiprocessing
_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4



def _get_experiment_matrix():
    experiments = []
    MODELS = [
        # "unsloth/gemma-3-270m",
        # "google/gemma-3-270m"
        "google/gemma-3-1b-pt",
        # "google/gemma-3-4b-it"
    ]

    DATASET_COMBINATIONS = [
        # ("chess", "math"),
        ("medical", "finance"),
        ("medical", "math"),
        ("medical", "coding"),
        ("finance", "math"),
        ("finance", "coding"),
        ("math", "coding"),
    ]
    print(DATASET_COMBINATIONS)

    print(set(DATASET_COMBINATIONS))

    _ = input("Press Enter to continue...")



    for model in MODELS:
        for client_0_dataset, client_1_dataset in DATASET_COMBINATIONS:
            experiment = {
                "model_name": model,
                "client_0_dataset": client_0_dataset,
                "client_1_dataset": client_1_dataset,
            }
            experiments.append(experiment)

    return experiments


def _generate_experiment_config(experiment_setting_dict, use_lora):
    config = ConfigManager.load_default_config()
    config["model_config"]["model_name"] = experiment_setting_dict["model_name"]
    config["dataset"]["client_0_dataset"] = experiment_setting_dict["client_0_dataset"]
    config["dataset"]["client_1_dataset"] = experiment_setting_dict["client_1_dataset"]
    if use_lora:
        config["lora_config"]["use_lora"] = True
    return config


def run_experiments():
    all_experiments = _get_experiment_matrix()
    for i, exp in enumerate(all_experiments):
        config = _generate_experiment_config(exp, use_lora=True)
        experiment_key = ConfigManager.generate_exp_key(config)
        metrics = {}
        print(f"[{i}/{len(all_experiments)}] Running: {experiment_key}")


        if CacheManager.experiment_is_complete(experiment_key):
            print("✅ Experiment already completed. Skipping...")
            continue

        start_time = time.time()
        metrics = run_fl_experiment(config)
        duration = time.time() - start_time

        CacheManager.consolidate_experiment(experiment_key, config, metrics)
        save_json(metrics, f"results/fl_train_metrics_{experiment_key}.json")
        print(f"✅ Completed in {duration:.1f}s")
        del metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n🎉 All {len(all_experiments)} experiments completed!")


if __name__ == "__main__":
    run_experiments()
