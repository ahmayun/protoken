from diskcache import Index
import os
import json
import re
from tqdm import tqdm
from peft import set_peft_model_state_dict
# from unsloth import FastLanguageModel

from src.utils.model import get_model_and_tokenizer


def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON data to: {json_path}")


def sanitize_key(name, slash_replacement="_", max_length=255):
    name = str(name).translate({ord("/"): slash_replacement, 0: None}).strip()
    if slash_replacement:
        rep = re.escape(slash_replacement)
        name = re.sub(fr"{rep}{{2,}}", slash_replacement,
                      name).strip(slash_replacement)

    return name[:max_length] if max_length else name


class CacheManager:
    TEMP_CLIENTS_MODELS_CACHE = "_storage/caches/clients_models_cache"
    TEMP_ROUNDS_STORAGE_CACHE = "_storage/caches/rounds_storage_cache"
    EXPERIMENT_CACHE = "_storage/caches/complete_experiment_cache"

    @staticmethod
    def _get_temp_clients_cache():
        return Index(CacheManager.TEMP_CLIENTS_MODELS_CACHE)

    @staticmethod
    def _get_temp_rounds_cache():
        return Index(CacheManager.TEMP_ROUNDS_STORAGE_CACHE)

    @staticmethod
    def _to_cuda_fp32(model):
        device = "cuda"
        return model.to(device=device)

    @staticmethod
    def save_client_trained_state(cid, client_training_state):
        print(f"Saving trained state for client {cid}...")
        cache = CacheManager._get_temp_clients_cache()
        cache[cid] = client_training_state
        print(
            f"Saved trained state for client {cid} to {CacheManager.TEMP_CLIENTS_MODELS_CACHE}")

    def save_round_state(round_num, global_model_state, clients_state):
        cache = CacheManager._get_temp_rounds_cache()
        cache[round_num] = {
            "global": global_model_state,
            "clients": clients_state
        }
        print(
            f"Saved round {round_num} state to {CacheManager.TEMP_ROUNDS_STORAGE_CACHE}")

    @staticmethod
    def get_clients_state():
        cache = CacheManager._get_temp_clients_cache()
        clients_state = {}
        for key in cache:
            clients_state[key] = cache[key]
        return clients_state

    @staticmethod
    def remove_clients_state():
        cache = CacheManager._get_temp_clients_cache()
        cache.clear()
        print(
            f"Removed all clients' states from {CacheManager.TEMP_CLIENTS_MODELS_CACHE}")

    @staticmethod
    def remove_rounds_state():
        cache = CacheManager._get_temp_rounds_cache()
        cache.clear()
        print(
            f"Removed all rounds' states from {CacheManager.TEMP_ROUNDS_STORAGE_CACHE}")

    @staticmethod
    def get_clients_state_count():
        cache = CacheManager._get_temp_clients_cache()
        return len(cache)

    @staticmethod
    def _load_model(config, state_dict):

        model, tokenizer = get_model_and_tokenizer(config)

        if hasattr(model, 'peft_config'):
            print("Global: LoRA model detected. Loading with LoRA layers.")
            set_peft_model_state_dict(model, state_dict)

            # # Access the TMPDIR environment variable
            # tmpdir = os.environ.get("TMPDIR")
            # model_path = '_storage/merged_model_16bit'
            # if tmpdir:
            #     model_path = f'{tmpdir}/merged_model_16bit'
            #     print(f'Model path set to Temp Directory: {model_path}')

            # model = model.cpu()
            # model.save_pretrained_merged(
            #     model_path, None, save_method="merged_16bit")
            # model, _ = FastLanguageModel.from_pretrained(model_name=model_path)
            # model = model.merge_and_unload()
            CacheManager._to_cuda_fp32(model)

        else:
            model.load_state_dict(state_dict, strict=True)

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
    def consolidate_experiment(exp_key, experiment_config, metrics):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        temp_rounds_cache = CacheManager._get_temp_rounds_cache()
        for round_id in tqdm(range(experiment_config["fl"]["num_rounds"]), desc="Merging rounds cache"):
            round_key = f"{exp_key}-round-{round_id}"
            experiment_cache[round_key] = temp_rounds_cache[round_id]

        experiment_cache[exp_key] = {
            "experiment_config": experiment_config,
            "experiment_metrics": metrics
        }
        print("==== Merge global and local model cache is complete. ====")

    @staticmethod
    def load_models_and_tokenizer_for_round(exp_key, round_id):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]
        round_key = f"{exp_key}-round-{round_id}"
        round_dict = experiment_cache[round_key]
        return CacheManager._load_models_and_tokenizer_from_round_dict(round_dict, exp_info["experiment_config"])

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
