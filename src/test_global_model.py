from src.fl_main import *
from diskcache import Index
import torch
from transformers import TextStreamer

def load_model_from_cache(round_num):
    cache = Index("_storage/model_cache")
    model, tokenizer = get_model_and_tokenizer()
    cached_data = cache[round_num]
    model.load_state_dict(cached_data["global_model"])
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), tokenizer, cached_data["metrics"]

def generate_response(model, tokenizer, dataset, sample_idx=10):
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

    print("\n[Generating response...]<<<<")
    
    generated = model.generate(
        **tokenizer(text, return_tensors="pt").to(model.device),
        max_new_tokens=125,
        temperature=1, 
        top_p=0.95, 
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    
    print(">>>>")
    print(f"\n\n\n> Actual result:\n{conversation[2]['content']}")
    
    return conversation[2]['content']

def test_model_round(round_num, sample_idx=10):
    print(f"\n{'='*60}")
    print(f"TESTING GLOBAL MODEL - ROUND {round_num}")
    print(f"{'='*60}")
    
    model, tokenizer, metrics = load_model_from_cache(round_num)
    print(f"Loaded model from round {round_num}")
    print(f"Round metrics: {metrics}")
    
    print(f"\n{'-'*40}")
    print("TESTING CHESS DATASET")
    print(f"{'-'*40}")
    chess_dataset = get_client_dataset("0")
    generate_response(model, tokenizer, chess_dataset, sample_idx)
    
    print(f"\n{'-'*40}")
    print("TESTING MATH DATASET")
    print(f"{'-'*40}")
    math_dataset = get_client_dataset("1")
    generate_response(model, tokenizer, math_dataset, sample_idx)
    
    

def test_rounds(rounds, sample_idx=10):
    for round_num in rounds:
        test_model_round(round_num, sample_idx)

def test_all_available_rounds(sample_idx=10):
    cache = Index("_storage/model_cache")
    available_rounds = list(cache.keys())
    available_rounds.sort()
    
    print(f"Available rounds: {available_rounds}")
    test_rounds(available_rounds, sample_idx)


if __name__ == "__main__":
    test_all_available_rounds(sample_idx=10)
    
