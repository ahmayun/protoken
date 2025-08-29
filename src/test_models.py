from src.fl_main import *
from src.test_global_model import generate_response
from diskcache import Index
import torch

def load_global_model_from_cache(round_num):
    cache = Index("_storage/model_cache")
    if round_num not in cache:
        return None, None, None
        
    model, tokenizer = get_model_and_tokenizer()
    cached_data = cache[round_num]
    model.load_state_dict(cached_data["global_model"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, cached_data["metrics"]

def load_all_client_models_from_cache(round_num):
    client_cache = Index("_storage/client_models")
    client_models = {}
    
    for key in client_cache:
        if key.endswith(f"_round_{round_num}"):
            client_id = key.split("_")[1]
            model_data = client_cache[key]
            
            model, tokenizer = get_model_and_tokenizer()
            model.load_state_dict(model_data["model_state_dict"])
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            
            client_models[client_id] = (model, tokenizer, model_data)
    
    return client_models

def test_round_comprehensive(round_num, sample_idx=10):
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL TESTING - ROUND {round_num}")
    print(f"{'='*60}")
    
    global_model, global_tokenizer, global_metrics = load_global_model_from_cache(round_num)
    client_models = load_all_client_models_from_cache(round_num)
    
    available_models = []

    available_models.extend([f"Client {cid}" for cid in sorted(client_models.keys())])  
    print(f"Available models: {', '.join(available_models)}")
    
    chess_dataset = get_client_dataset("0")
    math_dataset = get_client_dataset("1")
    
    
    print(f"\n{'*'*50}")
    print(f"GLOBAL MODEL - ROUND {round_num}")
    print(f"{'*'*50}")
    print(f"Metrics: {global_metrics}")
    
    print(f"\n{'-'*40}")
    print(f"GLOBAL MODEL - TESTING CHESS DATASET")
    print(f"{'-'*40}")
    generate_response(global_model, global_tokenizer, chess_dataset, sample_idx)
    
    print(f"\n{'-'*40}")
    print(f"GLOBAL MODEL - TESTING MATH DATASET")
    print(f"{'-'*40}")
    generate_response(global_model, global_tokenizer, math_dataset, sample_idx)
    
    for client_id in sorted(client_models.keys()):
        model, tokenizer, client_data = client_models[client_id]
        
        print(f"\n{'*'*50}")
        print(f"CLIENT {client_id} - ROUND {round_num}")
        print(f"{'*'*50}")
        print(f"Training metrics: {client_data.get('training_metrics', 'N/A')}")
        print(f"Dataset size: {client_data.get('dataset_size', 'N/A')}")
        
        print(f"\n{'-'*40}")
        print(f"CLIENT {client_id} - TESTING CHESS DATASET")
        print(f"{'-'*40}")
        generate_response(model, tokenizer, chess_dataset, sample_idx)
        
        print(f"\n{'-'*40}")
        print(f"CLIENT {client_id} - TESTING MATH DATASET")
        print(f"{'-'*40}")
        generate_response(model, tokenizer, math_dataset, sample_idx)

def test_all_available_rounds(sample_idx=10):
    # global_cache = Index("_storage/model_cache")
    
    # available_rounds = set()
    
    # for round_num in global_cache.keys():
    #     available_rounds.add(round_num)   
    
    # available_rounds = sorted(list(available_rounds))

    available_rounds = [5]
           
    print(f"Available rounds: {available_rounds}")
    
    for round_num in available_rounds:
        test_round_comprehensive(round_num, sample_idx)

if __name__ == "__main__":
    test_all_available_rounds(sample_idx=10)
