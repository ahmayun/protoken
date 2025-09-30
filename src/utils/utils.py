import unsloth
from diskcache import Index 
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

def get_model_and_tokenizer(config):
    model_config = config["model_config"]
    get_chat_template_name = config["chat_template"]
    model, tokenizer = FastModel.from_pretrained(**model_config)
    tokenizer = get_chat_template(
        tokenizer, chat_template=get_chat_template_name)
    return model, tokenizer


class CacheManager:
    TEMP_MODEL_CACHE = "_storage/caches/global_model_cache"
    TEMP_CLIENT_CACHE = "_storage/caches/client_models"
    EXPERIMENT_CACHE = "_storage/caches/complete_experiment_cache"
        
    @staticmethod
    def _get_temp_model_cache():
        return Index(CacheManager.TEMP_MODEL_CACHE)
    
    @staticmethod
    def _get_temp_client_cache():
        return Index(CacheManager.TEMP_CLIENT_CACHE)
    
    @staticmethod
    def _client_key(cid, round_num):
        return f"client_{cid}_round_{round_num}"
    
    @staticmethod
    def _load_temp_global_model(round_num):
        cache = Index(CacheManager.TEMP_MODEL_CACHE)
        cached_data = cache[round_num]
        return cached_data["global_model"], cached_data["metrics"]
    
    @staticmethod
    def _load_temp_client_models(round_num):
        cache = Index(CacheManager.TEMP_CLIENT_CACHE)
        client_models = {}
        for key in cache:
            if key.endswith(f"_round_{round_num}"):
                client_id = key.split("_")[1]
                client_models[client_id] = cache[key]
        return client_models
        
    @staticmethod
    def _to_cuda_fp32(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device=device, dtype=torch.float32)
    
    @staticmethod
    def _load_models_and_tokenizer_from_round_dict(round_dict, config):
        
        global_model, global_tokenizer = get_model_and_tokenizer(config)
        
        global_model_state_dict = round_dict["global"]["model"]
        global_model.load_state_dict(global_model_state_dict, strict=False)
        global_model = CacheManager._to_cuda_fp32(global_model)
        
        client_models = {}
        for client_id, client_data in round_dict["clients"].items():
            client_model, _ = get_model_and_tokenizer(config)
            client_model_state_dict = client_data["model_state_dict"]
            client_model.load_state_dict(client_model_state_dict, strict=False)
            client_model = CacheManager._to_cuda_fp32(client_model)
            client_models[client_id] = client_model
        
    
        print(f"==== Loaded global and {len(client_models)} client models from cache. ====")        
        return global_model, global_tokenizer, client_models
    
    @staticmethod
    def _load_experiment_round(exp_key, round_id):
        cache = Index(CacheManager.EXPERIMENT_CACHE)
        round_key = f"{exp_key}-round-{round_id}"
        return cache[round_key]
    
    @staticmethod
    def save_temp_global_model(round_num, model_state_dict, metrics):
        cache = CacheManager._get_temp_model_cache()
        cache[round_num] = {"global_model": model_state_dict, "metrics": metrics}
    
    @staticmethod  
    def save_temp_client_model(cid, round_num, model_data):
        cache = CacheManager._get_temp_client_cache()
        key = CacheManager._client_key(cid, round_num)
        cache[key] = model_data
    
    @staticmethod
    def consolidate_experiment(exp_key, experiment_config):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        for round_id in range(experiment_config["fl"]["num_rounds"]):
            round_key = f"{exp_key}-round-{round_id}"
            
            global_model_state_dict, global_model_metrics = CacheManager._load_temp_global_model(round_id)
            clients_models = CacheManager._load_temp_client_models(round_id)
            
            experiment_cache[round_key] = {
                "global": {"model": global_model_state_dict, "metrics": global_model_metrics}, 
                "clients": clients_models
            }
        
        experiment_cache[exp_key] = {
            "experiment_config": experiment_config
        }
        print("==== Merge global and local model cache is complete. ====")

    @staticmethod
    def load_models_and_tokenizer_for_round(exp_key, round_id):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        exp_info = experiment_cache[exp_key]        
        round_dict = CacheManager._load_experiment_round(exp_key, round_id)
        return CacheManager._load_models_and_tokenizer_from_round_dict(round_dict, exp_info["experiment_config"])
