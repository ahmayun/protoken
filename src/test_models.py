import unsloth
from diskcache import Index
import torch
from transformers import TextStreamer
from src.fl_train import get_model_and_tokenizer, get_client_dataset
from src.fl_prov import ProvTextGenerator, get_all_layers
from src.judge import llm_judge
from openai import OpenAI
import gc


def _to_cuda_fp32(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    print(">> Generated Response (Prediction) ->", end=" ")

    generated = model.generate(
        **tokenizer(text, return_tensors="pt").to(model.device),
        max_new_tokens=125,
        temperature=1,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    print(
        f"[[Actual Response (Ground Truth) ->  {conversation[2]['content']}]]")

    return conversation[2]['content']


def generate_response_with_provenance(model, tokenizer, dataset, sample_idx, client_models, terminators):
    print(
        f"\n\n{15*'-'} Input Index {sample_idx} Response Generation Provenance {15*'-'}")
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

    result = ProvTextGenerator.generate_text(
        model, client_models, tokenizer, text, terminators
    )
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    match = llm_judge(
        predicted=result['response'], actual=conversation[2]['content'], client=client)

    print(f">> Actual Response (Ground Truth) -> {conversation[2]['content']}")
    print(f">> Generated Response: {result['response']}")
    print(
        f">> LLM Judge Verdict Generated Vs Actual Response: {'Match' if match else 'No Match'}")
    print(f">> Provenance Analysis: {result['client2part']}")
    print(
        f">> Predicted Client: {max(result['client2part'], key=result['client2part'].get)}")

    result['is_generation_correct'] = match

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


def evaluate_provenance(global_model, global_tokenizer, dataset, sample_idxs, client_models, terminators, expected_client_id):
    prov_acc = []
    for sample_idx in sample_idxs:
        result = generate_response_with_provenance(
            global_model, global_tokenizer, dataset, sample_idx,
            client_models, terminators
        )
        if result['is_generation_correct']:
            traced_client_id = max(
                result['client2part'], key=result['client2part'].get)
            prov_acc.append(traced_client_id == expected_client_id)
    return prov_acc


def rounds_provenance(rounds, sample_idxs):
    datasets = [
        ("chess", get_client_dataset("0"), "0"),
        ("math", get_client_dataset("1"), "1"),
    ]

    for round_num in rounds:

        print(f"\n{'='*60}")
        print(
            f"PROVENANCE ANALYSIS - ROUND {round_num}")
        print(f"{'='*60}")

        global_model, global_tokenizer, _ = load_global_model_from_cache(
            round_num)
        client_models_data = load_all_client_models_from_cache(round_num)
        client_models = {cid: model for cid,
                         (model, _, _) in client_models_data.items()}
        terminators = [global_tokenizer.eos_token_id]

        for dataset_name, dataset, expected_client_id in datasets:
            print(f"\n{'-'*40}")
            print(f"PROVENANCE - {dataset_name.upper()} DATASET")
            print(f"{'-'*40}")
            prov_acc_list =  evaluate_provenance(global_model, global_tokenizer, dataset, sample_idxs,
                                client_models, terminators, expected_client_id)
            print(f">> Provenance Accuracy: {sum(prov_acc_list)}/{len(prov_acc_list)} = {100.0 * sum(prov_acc_list)/len(prov_acc_list) if len(prov_acc_list) > 0 else 0.0:.2f}%")
        
        _ = [m.cpu() for m in client_models.values()]
        global_model.cpu()


        del client_models
        del global_model
        torch.cuda.empty_cache()
        gc.collect()


def test_all_available_rounds(sample_idx=10):
    # global_cache = Index("_storage/model_cache")

    # available_rounds = set()

    # for round_num in global_cache.keys():
    #     available_rounds.add(round_num)

    # available_rounds = sorted(list(available_rounds))

    # print(f"Layers: {get_all_layers(global_model)}")
    # print(f"Model {model}")
    # for name, mode in global_model.named_modules():
    #     print(f"Layer: {name}, Type: {type(mode)}")
    # _ = input("Press Enter to continue...")

    available_rounds = [1]

    print(f"Available rounds: {available_rounds}")

    for round_num in available_rounds:
        test_round_comprehensive(round_num, sample_idx)


if __name__ == "__main__":
    rounds_provenance(rounds=list(range(1, 11)), sample_idxs=list(range(20)))

    # test_all_available_rounds(sample_idx=10)
