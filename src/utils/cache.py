from diskcache import Index
from peft import set_peft_model_state_dict
from src.fl.model import get_model_and_tokenizer
from pathlib import Path
import os
import time
from joblib import Parallel, delayed    


# # tmpfs = os.getenv('TMPFS', '/tmp')    # default value will be /tmp. Change as needed.
# tmpfs_dir = os.environ['TMPFS']  # #TMPDIR very fast io on cluster
# if tmpfs_dir is None or not Path(tmpfs_dir).exists():
#     raise ValueError(
#         f"TMPFS directory {tmpfs_dir} does not exist. Please set TMPFS environment variable to a valid path.")


def _load_model(config, state_dict):

    model, _ = get_model_and_tokenizer(config)

    if hasattr(model, 'peft_config'):
        print("Global: LoRA model detected. Loading with LoRA layers.")
        set_peft_model_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict, strict=True)

    return model.to("cuda")




def _get_client_model_from_cache(round_key, client_key, config):
    experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
    print(f"Loading model for {client_key} ... ")
    model = _load_model(config, experiment_cache[client_key]['model'])
    
    client_id = client_key.split(f"{round_key}-client-")[1]
    print(f"Client {client_id} model loaded.")
    return client_id, model



def _load_rounds_dict(round_key, config):
    experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
    global_model = _load_model(config,experiment_cache[f"{round_key}-global"]['model'])


    clients_models_keys = [key for key in experiment_cache.keys() if key.startswith(f"{round_key}-client-")]
    
    start_time = time.perf_counter()  # <-- Start timer
    results = [_get_client_model_from_cache(round_key, client_key, config) for client_key in clients_models_keys]
    # results = Parallel(n_jobs=2)(delayed(_get_client_model_from_cache)(round_key, client_key, config) for client_key in clients_models_keys)
    
    end_time = time.perf_counter()  # <-- Stop timer
    print(f"--- Took: {end_time - start_time:.2f} seconds to load models ---")

    client2model = {client_id: model for client_id, model in results}
 
    assert len(client2model) > 0, "No client states found in cache."

    return global_model, client2model


        

class CacheManager:
    EXPERIMENT_CACHE = os.environ.get(
        "GENFL_EXPERIMENT_CACHE",
        "/scratch/ahmad35/_storage/caches/complete_experiment_cache-3",
    )



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



    # ================provenance methods
    @staticmethod
    def get_completed_experiments_keys():
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        keys = []
        for key in experiment_cache:
            if '-client-' in key or '-global' in key:
                continue            
            keys.append(key)
        return keys


    

    @staticmethod
    def load_models_and_tokenizer_for_round(exp_key, round_id):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]
        round_key = f"{exp_key}-round-{round_id}"
        print(f"Loading models for {round_key} ... ")
        round_dict = _load_rounds_dict(round_key, exp_info["experiment_config"])
        return round_dict

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
