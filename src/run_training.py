import time
import torch
import gc
import copy
from src.fl.simulation import run_fl_experiment
from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.config import ConfigManager


def _get_experiment_matrix():
    experiments = []
    fl_config = {
        "num_rounds": 10,
        "num_clients": 25,
        "clients_per_round": 4
    }

    total_gpus = 2
    total_cpus =  12
    client_resources =  {
            "num_cpus": 4,
            "num_gpus": 2
    }

    # "google/gemma-3-1b-pt",
    # "google/gemma-3-4b-it"
    # "google/gemma-3-1b-it",
    # "Qwen/Qwen3-0.6B"
# 'google/gemma-3-270m'
    models = [
         {'model_name': "google/gemma-3-270m"},
        {'model_name': "google/gemma-3-1b-pt"},
        {'model_name': "meta-llama/Llama-3.2-1B"},

    ]

    #   "dataset": {
    #         "samples_per_client": 512,
    #         "test_dataset_size": 512,
    #         "classes_per_client": 1,
    #         # "labels_to_keep": ['medical', 'finance'], # 93
    #         # "labels_to_keep": ['medical', 'math'], # 86
    #         # "labels_to_keep": ['finance', 'math'], # 80. accurate
    #         # 'labels_to_keep': ['chess', 'math'], # 77
    #         # 'labels_to_keep': ['math', 'coding'], #48
    #         "labels_to_keep": ['medical', 'finance', 'math']
    #         # "labels_to_keep": ['medical', 'finance']

    base_ds_config = {
        "samples_per_client": 512,
        "test_dataset_size": 512,
        "classes_per_client": 1,
        "labels_to_keep": None
    }

    all_ds_paisr = [['medical', 'finance', 'math'], ['medical', 'finance',], ['medical', 'math']]
    all_ds_configs =[]
    for labels in all_ds_paisr:
        ds_config =  copy.deepcopy(base_ds_config)
        ds_config['labels_to_keep'] = labels
        all_ds_configs.append(ds_config)
    

    for model_config in models:
        for dataset_config in all_ds_configs:    

            experiment = {
                'model_config': model_config,
                'fl': fl_config,
                'dataset': dataset_config,
                'total_gpus': total_gpus,
                'total_cpus': total_cpus,
                'client_resources': client_resources
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

    if CacheManager.experiment_is_complete(experiment_key):
        print("✅ Experiment already completed. Skipping...")
        return
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
