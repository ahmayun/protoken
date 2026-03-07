import argparse
import os
import time
import torch
import gc
import copy
from src.fl.simulation import run_fl_experiment
from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.config import ConfigManager
from pathlib import Path

# Reduce CUDA fragmentation (can help with large allocations)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Short names (from reproduce.sh) -> HuggingFace model id
MODEL_SHORT_TO_ID = {
    "gemma": "google/gemma-3-270m-it",
    "smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}


def _get_experiment_matrix(model=None, dataset=None, rounds=None):
    """Build experiment matrix for scalability (55 clients). Defaults hardcoded; args override."""
    experiments = []
    num_rounds = 16 if rounds is None else int(rounds)
    fl_config = {
        "num_rounds": num_rounds,
        "num_clients": 55,
        "clients_per_round": 10,
    }

    total_gpus = 1
    total_cpus = 8
    client_resources = {"num_cpus": 2, "num_gpus": 1}

    if model is None:
        models = [{"model_name": "HuggingFaceTB/SmolLM2-360M-Instruct"}]
    else:
        model_id = MODEL_SHORT_TO_ID.get(model.strip().lower(), model)
        models = [{"model_name": model_id}]

    base_ds_config = {
        "samples_per_client": 200,
        "test_dataset_size": 1024,
        "classes_per_client": None,
        "labels_to_keep": None,
        "partition_strategy": "iid",
        "inject_backdoor": True,
        "backdoor_clients": [f"{c}" for c in range(25)],
    }

    if dataset is None:
        all_ds_pairs = [["medical"]]
    else:
        all_ds_pairs = [[d.strip() for d in dataset.split(",")]]

    all_ds_configs = []
    for labels in all_ds_pairs:
        ds_config = copy.deepcopy(base_ds_config)
        ds_config["labels_to_keep"] = labels
        all_ds_configs.append(ds_config)

    for model_config in models:
        for dataset_config in all_ds_configs:
            experiment = {
                "model_config": model_config,
                "fl": fl_config,
                "dataset": dataset_config,
                "total_gpus": total_gpus,
                "total_cpus": total_cpus,
                "client_resources": client_resources,
            }
            experiments.append(experiment)

    return experiments


def _generate_experiment_config(experiment_setting_dict, use_lora, epochs):
    config = ConfigManager.load_default_config()
    for k, v in experiment_setting_dict.items():
        config[k] = v
    config["use_lora"] = use_lora
    config["sft_config_args"]["num_train_epochs"] = epochs
    # Lower peak memory: smaller batch + grad accum (effective batch ~32)
    config["sft_config_args"]["per_device_train_batch_size"] = 8
    config["sft_config_args"]["gradient_accumulation_steps"] = 4
    return config


def _is_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in (exc.args[0] or "").lower():
        return True
    return False


def _reduce_memory_config(config):
    """Halve batch size and double grad accum to keep effective batch size."""
    cfg = config["sft_config_args"]
    batch = max(1, cfg["per_device_train_batch_size"] // 2)
    accum = cfg["gradient_accumulation_steps"] * 2
    cfg["per_device_train_batch_size"] = batch
    cfg["gradient_accumulation_steps"] = accum
    return config


def single_exp_run(config, exp_dir, max_oom_retries=3):
    experiment_key = ConfigManager.generate_exp_key(config)

    print(f"Running: {experiment_key}")

    if CacheManager.experiment_is_complete(experiment_key):
        print("✅ Experiment already completed. Skipping...")
        return

    torch.cuda.empty_cache()
    gc.collect()

    config = copy.deepcopy(config)
    last_error = None
    for attempt in range(max_oom_retries):
        try:
            start_time = time.time()
            metrics = run_fl_experiment(experiment_key, config)
            duration = time.time() - start_time

            CacheManager.consolidate_experiment(experiment_key, config, metrics)
            save_json({"metrics": metrics, "config": config}, exp_dir / f"t_{experiment_key}.json")
            print(f"✅ Completed in {duration:.1f}s")
            del metrics
            torch.cuda.empty_cache()
            gc.collect()
            return
        except Exception as e:
            if _is_oom_error(e):
                last_error = e
                print(f"⚠️ OOM on attempt {attempt + 1}/{max_oom_retries}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                if attempt < max_oom_retries - 1:
                    config = _reduce_memory_config(config)
                    print(
                        f"   Retrying with batch_size={config['sft_config_args']['per_device_train_batch_size']} "
                        f"grad_accum={config['sft_config_args']['gradient_accumulation_steps']}..."
                    )
            else:
                raise
    raise last_error


def run_experiments(output_dir, model=None, dataset=None, rounds=None):
    experiment_dir = Path(output_dir) / "scalability" / "json"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_experiments = _get_experiment_matrix(model=model, dataset=dataset, rounds=rounds)
    epochs = 1
    for i, exp in enumerate(all_experiments):
        print(f"{i} Experiment Key: {ConfigManager.generate_exp_key(_generate_experiment_config(exp, use_lora=True, epochs=epochs))}")

    for i, exp in enumerate(all_experiments):
        config = _generate_experiment_config(exp, use_lora=False, epochs=epochs)
        print(f"Experiment [{i}/{len(all_experiments)}]")
        single_exp_run(config, experiment_dir)

    print(f"\n🎉 All {len(all_experiments)} experiments completed!")


def _parse_args():
    p = argparse.ArgumentParser(description="Train with backdoor for scalability (Fig 6 & 7, 55 clients).")
    p.add_argument("--model", default=None, help="Model short name (gemma|smollm|llama|qwen) or HuggingFace id.")
    p.add_argument("--dataset", default=None, help="Dataset name(s), comma-separated (e.g. medical). Default: medical.")
    p.add_argument("--rounds", type=int, default=None, help="Number of FL rounds. Default: 16.")
    p.add_argument("--output_dir", required=True, help="Output root; creates scalability/json under it.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiments(
        output_dir=args.output_dir,
        model=args.model,
        dataset=args.dataset,
        rounds=args.rounds,
    )
