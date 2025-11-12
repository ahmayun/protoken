import time
import torch
import gc
import copy
from src.fl.simulation import run_fl_experiment
from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.config import ConfigManager
from pathlib import Path


def _get_experiment_matrix():
    experiments = []
    fl_config = {
        "num_rounds": 16,
        "num_clients": 55,
        "clients_per_round": 10
    }

    total_gpus = 1
    total_cpus = 8
    client_resources = {
        "num_cpus": 2,
        "num_gpus": 1
    }

        # {'model_name': "google/gemma-3-1b-it"},


    models = [
        {'model_name': "google/gemma-3-270m-it"},
        {'model_name': "Qwen/Qwen2.5-0.5B-Instruct"},

        
        # {'model_name': "HuggingFaceTB/SmolLM2-360M-Instruct"},

        # {'model_name': "meta-llama/Llama-3.2-1B-Instruct"},
    ]

    base_ds_config = {
        "samples_per_client": 200,
        "test_dataset_size": 1024,
        "classes_per_client": None,
        "labels_to_keep": None,
        "partition_strategy": "iid",  # "pathological" | "iid"
        'inject_backdoor': True,
        "backdoor_clients": [f"{c}" for c in range(25)], 
    }

    # all_ds_paisr = [['medical', 'finance', 'math'],['medical', 'finance', 'math', 'chess'],['medical', 'finance',], ['medical', 'math']]

    all_ds_paisr = [
        # ['finance'],
        # ['math'],
        # ['medical'],
        ['coding']
    ]  # , ['math'], ['medical','finance', 'math']]
    all_ds_configs = []
    for labels in all_ds_paisr:
        ds_config = copy.deepcopy(base_ds_config)
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


def single_exp_run(config, exp_dir):
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
              exp_dir/f"t_{experiment_key}.json")
    print(f"✅ Completed in {duration:.1f}s")
    del metrics
    torch.cuda.empty_cache()
    gc.collect()


def run_experiments():
    experiment_dir = Path("results/train/backdoor/scalability/json/")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_experiments = _get_experiment_matrix()
    epochs = 1
    for i, exp in enumerate(all_experiments):
        print(f"{i} Experiment Key: {ConfigManager.generate_exp_key(_generate_experiment_config(exp, use_lora=True, epochs=epochs))}")

    _ = input("Only for test. Press enter to continue.")

    for i, exp in enumerate(all_experiments):
        config = _generate_experiment_config(
            exp, use_lora=False, epochs=epochs)

        print(f"Experiment [{i}/{len(all_experiments)}]")
        single_exp_run(config, experiment_dir)

    # for i, exp in enumerate(all_experiments):
    #     config = _generate_experiment_config(exp, use_lora=True, epochs=epochs)
    #     print(f"Experiment [{i}/{len(all_experiments)}]")
    #     single_exp_run(config)

    print(f"\n🎉 All {len(all_experiments)} experiments completed!")


if __name__ == "__main__":
    run_experiments()
