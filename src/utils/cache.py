from diskcache import Index
from peft import set_peft_model_state_dict
from src.fl.model import get_model_and_tokenizer
from pathlib import Path
import os
import time


# # tmpfs = os.getenv('TMPFS', '/tmp')    # default value will be /tmp. Change as needed.
# tmpfs_dir = os.environ['TMPFS']  # #TMPDIR very fast io on cluster
# if tmpfs_dir is None or not Path(tmpfs_dir).exists():
#     raise ValueError(
#         f"TMPFS directory {tmpfs_dir} does not exist. Please set TMPFS environment variable to a valid path.")


class CacheManager:
    # print(f"Using TMPFS directory for caches: {tmpfs_dir}")
    EXPERIMENT_CACHE = "/scratch/waris/_storage/caches/complete_experiment_cache"

    @staticmethod
    def _to_cuda_fp32(model):
        device = "cuda"
        return model.to(device=device)

    @staticmethod
    def load_experiment_configuration(exp_key):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]
        return exp_info["experiment_config"]

    @staticmethod
    def load_training_metrics(exp_key):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]
        return exp_info["experiment_metrics"]

    @staticmethod
    def set_provenance_results(exp_key, prov_results):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]
        exp_info['provenance_results'] = prov_results
        experiment_cache[exp_key] = exp_info
        print(f">> {exp_key} updated with provenance results.")

    @staticmethod
    def experiment_is_complete(exp_key):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        return exp_key in experiment_cache

    @staticmethod
    def get_completed_experiments_keys():
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        keys = []
        for key in experiment_cache:
            if key.find('-round-') == -1:  # exclude round-specific keys
                keys.append(key)
        return keys

    # ================provenance methods
    @staticmethod
    def _load_model(config, state_dict):

        model, tokenizer = get_model_and_tokenizer(config)

        if hasattr(model, 'peft_config'):
            print("Global: LoRA model detected. Loading with LoRA layers.")
            set_peft_model_state_dict(model, state_dict)
        else:
            model.load_state_dict(state_dict, strict=True)

        CacheManager._to_cuda_fp32(model)
        return model

    @staticmethod
    def _load_models_and_tokenizer_from_round_dict(round_dict, config):
        global_model_state_dict = round_dict["global"]["model"]
        global_model = CacheManager._load_model(
            config, global_model_state_dict)
        client_models = {}
        for client_id, client_data in round_dict["clients"].items():
            client_model_state_dict = client_data["model"]
            client_models[client_id] = CacheManager._load_model(
                config, client_model_state_dict)

        print(
            f"==== Loaded global and {len(client_models)} client models from cache. ====")
        return global_model, client_models

    @staticmethod
    def _load_rounds_dict(round_key):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        global_state_dict = experiment_cache[f"{round_key}-global"]
        clients_state_dict = {}
        for key in experiment_cache.keys():
            if not key.startswith(f"{round_key}-client-"):
                continue
            client_id = key.split(f"{round_key}-client-")[1]
            clients_state_dict[client_id] = experiment_cache[key]
            print(f"{key} state loaded.")

        return {"global": global_state_dict, "clients": clients_state_dict}

    @staticmethod
    def load_models_and_tokenizer_for_round(exp_key, round_id):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]
        round_key = f"{exp_key}-round-{round_id}"
        round_dict = CacheManager._load_rounds_dict(round_key)
        return CacheManager._load_models_and_tokenizer_from_round_dict(round_dict, exp_info["experiment_config"])

    # =============== Training Related ========================

    @staticmethod
    def save_client_trained_state(key, client_training_state):
        start = time.time()
        client_training_state['client_key_id'] = key
        print(
            f"Saving trained state for client {key}  ...")

        cache = Index(CacheManager.EXPERIMENT_CACHE)
        cache[key] = client_training_state
        print(
            f"Saved trained state for client {key} in {time.time()-start:.2f} seconds.")

    @staticmethod
    def save_global_state(exp_key, round_id, global_state):
        start = time.time()
        global_state['round'] = round_id
        global_state_key = f"{exp_key}-round-{round_id}-global"
        print(f"Saving global state for round {global_state_key} ...")

        cache = Index(CacheManager.EXPERIMENT_CACHE)
        cache[global_state_key] = global_state

        print(
            f"Saved global state for round {global_state_key} in {time.time()-start:.2f} seconds.")

    @staticmethod
    def consolidate_experiment(exp_key, experiment_config, metrics):
        start = time.time()
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        experiment_cache[exp_key] = {
            "experiment_config": experiment_config,
            "experiment_metrics": metrics
        }
        duration = time.time() - start
        print(
            f"==== Merge global and local model cache is complete in {duration} seconds. ====")
    
    @staticmethod
    def clear_training_with_key(exp_key):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        keys_to_delete = []
        for key in experiment_cache.keys():
            if key.startswith(exp_key):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del experiment_cache[key]
            print(f"Deleted cache key: {key}")

        print(f"Cleared all cache entries for experiment key: {exp_key}")
