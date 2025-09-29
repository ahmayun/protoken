from diskcache import Index 
import torch

class CacheManager:
    TEMP_MODEL_CACHE = "_storage/model_cache"
    TEMP_CLIENT_CACHE = "_storage/client_models" 
    EXPERIMENT_CACHE = "_storage/complete_experiment_cache"
    
    @staticmethod
    def load_experiment_round(exp_key, round_id):
        cache = Index(CacheManager.EXPERIMENT_CACHE)
        round_key = f"{exp_key}-round-{round_id}"
        return cache[round_key]
    
    @staticmethod
    def get_temp_model_cache():
        return Index(CacheManager.TEMP_MODEL_CACHE)
    
    @staticmethod
    def get_temp_client_cache():
        return Index(CacheManager.TEMP_CLIENT_CACHE)
    
    @staticmethod
    def client_key(cid, round_num):
        return f"client_{cid}_round_{round_num}"
    
    @staticmethod
    def load_temp_global_model(round_num):
        cache = Index(CacheManager.TEMP_MODEL_CACHE)
        cached_data = cache[round_num]
        return cached_data["global_model"], cached_data["metrics"]
    
    @staticmethod
    def load_temp_client_models(round_num):
        cache = Index(CacheManager.TEMP_CLIENT_CACHE)
        client_models = {}
        for key in cache:
            if key.endswith(f"_round_{round_num}"):
                client_id = key.split("_")[1]
                client_models[client_id] = cache[key]
        return client_models
    
    @staticmethod
    def save_temp_global_model(round_num, model_state_dict, metrics):
        cache = CacheManager.get_temp_model_cache()
        cache[round_num] = {"global_model": model_state_dict, "metrics": metrics}
    
    @staticmethod  
    def save_temp_client_model(cid, round_num, model_data):
        cache = CacheManager.get_temp_client_cache()
        key = CacheManager.client_key(cid, round_num)
        cache[key] = model_data
    
    @staticmethod
    def consolidate_experiment(exp_key, num_rounds):
        experiment_cache = Index(CacheManager.EXPERIMENT_CACHE)
        for round_id in range(num_rounds):
            round_key = f"{exp_key}-round-{round_id}"
            
            global_model_state_dict, global_model_metrics = CacheManager.load_temp_global_model(round_id)
            clients_models = CacheManager.load_temp_client_models(round_id)
            
            experiment_cache[round_key] = {
                "global": {"model": global_model_state_dict, "metrics": global_model_metrics}, 
                "clients": clients_models
            }
        print("==== Merge global and local model cache is complete. ====")
    
    @staticmethod
    def to_cuda_fp32(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device=device, dtype=torch.float32)
    
    @staticmethod
    def load_models_and_tokenizer_from_round_dict(round_dict):
        from src.fl_train import get_model_and_tokenizer
        
        global_model, global_tokenizer = get_model_and_tokenizer()
        
        global_model_state_dict = round_dict["global"]["model"]
        global_model.load_state_dict(global_model_state_dict, strict=False)
        global_model = CacheManager.to_cuda_fp32(global_model)
        
        client_models = {}
        for client_id, client_data in round_dict["clients"].items():
            client_model, _ = get_model_and_tokenizer()
            client_model_state_dict = client_data["model_state_dict"]
            client_model.load_state_dict(client_model_state_dict, strict=False)
            client_model = CacheManager.to_cuda_fp32(client_model)
            client_models[client_id] = client_model
        
        return global_model, global_tokenizer, client_models
    
    @staticmethod
    def load_models_and_tokenizer_for_round(exp_key, round_id):
        round_dict = CacheManager.load_experiment_round(exp_key, round_id)
        return CacheManager.load_models_and_tokenizer_from_round_dict(round_dict)
