from src.fl_main import *
from src.test_global_model import generate_response
from diskcache import Index
import torch


def load_client_model_from_cache(client_id, round_num):
    client_cache = Index("_storage/client_models")
    key = f"client_{client_id}_round_{round_num}"

    if key not in client_cache:
        return None, None, None

    model_data = client_cache[key]
    model, tokenizer = get_model_and_tokenizer()
    model.load_state_dict(model_data["model_state_dict"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    return model, tokenizer, model_data


def test_client_round(round_num, sample_idx=10):
    print(f"\n{'='*60}")
    print(f"TESTING CLIENT MODELS - ROUND {round_num}")
    print(f"{'='*60}")

    client_cache = Index("_storage/client_models")
    round_clients = []

    for key in client_cache:
        if key.endswith(f"_round_{round_num}"):
            client_id = key.split("_")[1]
            round_clients.append(client_id)

    if not round_clients:
        print(f"No client models found for round {round_num}")
        return

    round_clients.sort()
    print(f"Found clients: {round_clients}")

    chess_dataset = get_client_dataset("0")
    math_dataset = get_client_dataset("1")

    for client_id in round_clients:
        model, tokenizer, client_data = load_client_model_from_cache(
            client_id, round_num)

        if model is None:
            print(f"Failed to load client {client_id} model")
            continue

        print(f"\n{'*'*50}")
        print(f"CLIENT {client_id} - ROUND {round_num}")
        print(
            f"Training metrics: {client_data.get('training_metrics', 'N/A')}")
        print(f"Dataset size: {client_data.get('dataset_size', 'N/A')}")
        print(f"{'*'*50}")

        print(f"\n{'-'*40}")
        print(f"CLIENT {client_id} - TESTING CHESS DATASET")
        print(f"{'-'*40}")
        generate_response(model, tokenizer, chess_dataset, sample_idx)

        print(f"\n{'-'*40}")
        print(f"CLIENT {client_id} - TESTING MATH DATASET")
        print(f"{'-'*40}")
        generate_response(model, tokenizer, math_dataset, sample_idx)


def test_all_available_rounds(sample_idx=10):
    client_cache = Index("_storage/client_models")
    available_rounds = set()

    for key in client_cache:
        if "_round_" in key:
            round_num = int(key.split("_round_")[1])
            available_rounds.add(round_num)

    available_rounds = sorted(list(available_rounds))

    if not available_rounds:
        print("No client models found in cache")
        return

    print(f"Available rounds: {available_rounds}")

    for round_num in available_rounds:
        test_client_round(round_num, sample_idx)


if __name__ == "__main__":
    test_all_available_rounds(sample_idx=10)
