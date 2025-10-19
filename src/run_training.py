import time
import torch
import gc
from src.fl.simulation import run_fl_experiment
from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.config import ConfigManager


def _get_experiment_matrix():
    experiments = []
    fl_config = {
        "num_rounds": 5,
        "num_clients": 25,
        "clients_per_round": 3
    }

    total_gpus = 2

    # "google/gemma-3-1b-pt",
    # "google/gemma-3-4b-it"
    # "google/gemma-3-1b-it",
    # "Qwen/Qwen3-0.6B"

    models = [
        {'model_name': "google/gemma-3-1b-pt"}
    ]

    dataset_config = {
        "samples_per_client": 512,
        "test_dataset_size": 512,
        "classes_per_client": 1,
        "labels_to_keep": ['medical', 'finance', 'math']
    }

    for model_config in models:
        experiment = {
            'model_config': model_config,
            'fl': fl_config,
            'dataset': dataset_config,
            'total_gpus': total_gpus
        }
        experiments.append(experiment)

    return experiments


def _generate_experiment_config(experiment_setting_dict, use_lora, epochs):
    config = ConfigManager.load_default_config()
    for k, v in experiment_setting_dict.items():
        config[k] = v
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
    metrics = run_fl_experiment(experiment_key, config)
    duration = time.time() - start_time

    CacheManager.consolidate_experiment(experiment_key, config, metrics)
    save_json({"metrics": metrics, 'config': config},
              f"results/fl_train_metrics_{experiment_key}.json")
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
        config = _generate_experiment_config(
            exp, use_lora=False, epochs=epochs)
        print(f"Experiment [{i}/{len(all_experiments)}]")
        single_exp_run(config)

    # for i, exp in enumerate(all_experiments):
    #     config = _generate_experiment_config(exp, use_lora=True, epochs=epochs)
    #     print(f"Experiment [{i}/{len(all_experiments)}]")
    #     single_exp_run(config)

    print(f"\n🎉 All {len(all_experiments)} experiments completed!")


if __name__ == "__main__":
    run_experiments()
