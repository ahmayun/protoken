import unsloth
from diskcache import Index
import torch
from transformers import TextStreamer
from src.fl_train import get_model_and_tokenizer, get_client_dataset
from src.fl_prov import ProvTextGenerator, get_all_layers

def _to_cuda_fp32(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # .to(dtype=...) casts all parameters & buffers; keeps device if already set
    return model.to(device=device, dtype=torch.float32)


def generate_response(model, tokenizer, dataset, sample_idx=10):
    conversation = dataset['conversations'][sample_idx]

    messages = [
        {'role': conversation[0]['role'],
            'content': conversation[0]['content']},
        {"role": conversation[1]['role'],
            'content': conversation[1]['content']}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')

    print(">> Generated Response (Prediction) ->", end= " ")

    generated = model.generate(
        **tokenizer(text, return_tensors="pt").to(model.device),
        max_new_tokens=125,
        temperature=1,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    print(f"[[Actual Response (Ground Truth) ->  {conversation[2]['content']}]]")

    return conversation[2]['content']

def generate_response_with_provenance(model, tokenizer, dataset, sample_idx, client_models, terminators):
    conversation = dataset['conversations'][sample_idx]
    
    messages = [
        {'role': conversation[0]['role'], 'content': conversation[0]['content']},
        {"role": conversation[1]['role'], 'content': conversation[1]['content']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')

    

    result = ProvTextGenerator.generate_text(
        model, client_models, tokenizer, text, terminators
    )
    
    print(f">> Provenance Analysis: {result['client2part']}")
    print(f">> Generated Response: {result['response']}")
    print(f"[[Actual Response (Ground Truth) -> {conversation[2]['content']}]]")
    
    return result

def load_global_model_from_cache(round_num):
    cache = Index("_storage/model_cache")
    model, tokenizer = get_model_and_tokenizer()
    cached_data = cache[round_num]
    model.load_state_dict(cached_data["global_model"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = _to_cuda_fp32(model)
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
            model = _to_cuda_fp32(model)
            client_models[client_id] = (model, tokenizer, model_data)

    return client_models

def test_round_comprehensive(round_num, sample_idx=10):
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL TESTING - ROUND {round_num}")
    print(f"{'='*60}")

    global_model, global_tokenizer, global_metrics = load_global_model_from_cache(
        round_num)
    client_models = load_all_client_models_from_cache(round_num)

    available_models = []

    available_models.extend(
        [f"Client {cid}" for cid in sorted(client_models.keys())])
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
    generate_response(global_model, global_tokenizer,
                      chess_dataset, sample_idx)

    print(f"\n{'-'*40}")
    print(f"GLOBAL MODEL - TESTING MATH DATASET")
    print(f"{'-'*40}")
    generate_response(global_model, global_tokenizer, math_dataset, sample_idx)

    for client_id in sorted(client_models.keys()):
        model, tokenizer, client_data = client_models[client_id]

        print(f"\n{'*'*50}")
        print(f"CLIENT {client_id} - ROUND {round_num}")
        print(f"{'*'*50}")
        # print(
        #     f"Training metrics: {client_data.get('training_metrics', 'N/A')}")
        
        print(f"Dataset size: {client_data.get('dataset_size', 'N/A')}")

        print(f"\n{'-'*40}")
        print(f"CLIENT {client_id} - TESTING CHESS DATASET")
        print(f"{'-'*40}")
        generate_response(model, tokenizer, chess_dataset, sample_idx)

        print(f"\n{'-'*40}")
        print(f"CLIENT {client_id} - TESTING MATH DATASET")
        print(f"{'-'*40}")
        generate_response(model, tokenizer, math_dataset, sample_idx)



def test_rounds_batch_optimized(rounds, sample_idxs):
    chess_dataset = get_client_dataset("0")
    math_dataset = get_client_dataset("1")
    
    for round_num in rounds:


        global_model, global_tokenizer, _ = load_global_model_from_cache(round_num)
        client_models_data = load_all_client_models_from_cache(round_num)
        client_models = {cid: model for cid, (model, _, _) in client_models_data.items()}
        terminators = [global_tokenizer.eos_token_id]

        # print(f"Model {model}")

        for name, mode in global_model.named_modules():
            print(f"Layer: {name}, Type: {type(mode)}")

        print(f"Layers: {get_all_layers(global_model)}")

        _ = input("Press Enter to continue...")
        
        
        
        for sample_idx in sample_idxs:
            print(f"\n{'='*60}")
            print(f"PROVENANCE ANALYSIS - ROUND {round_num} - SAMPLE {sample_idx}")
            print(f"{'='*60}")
            
            print(f"\n{'-'*40}")
            print(f"PROVENANCE - CHESS DATASET")
            print(f"{'-'*40}")
            generate_response_with_provenance(
                global_model, global_tokenizer, chess_dataset, sample_idx,
                client_models, terminators
            )
            
            print(f"\n{'-'*40}")
            print(f"PROVENANCE - MATH DATASET")
            print(f"{'-'*40}")
            generate_response_with_provenance(
                global_model, global_tokenizer, math_dataset, sample_idx,
                client_models, terminators
            )

def test_all_available_rounds(sample_idx=10):
    # global_cache = Index("_storage/model_cache")

    # available_rounds = set()

    # for round_num in global_cache.keys():
    #     available_rounds.add(round_num)

    # available_rounds = sorted(list(available_rounds))

    available_rounds = [1]

    print(f"Available rounds: {available_rounds}")

    for round_num in available_rounds:
        test_round_comprehensive(round_num, sample_idx)

if __name__ == "__main__":
    # test_round_with_provenance(round_num=1, sample_idx=10)
    test_rounds_batch_optimized(rounds= [9,10], sample_idxs=[10, 15, 100, 200, 400])


    # test_all_available_rounds(sample_idx=10)
