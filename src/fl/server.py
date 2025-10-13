import flwr as fl
from peft import get_peft_model_state_dict
import torch
import gc
from flwr.common import ndarrays_to_parameters


from src.utils.utils import CacheManager
from src.utils.model import ModelUtils, evaluate_llm


def average_dicts(metrics_dict):
    sums = {}
    for d in metrics_dict.values():
        for key, value in d.items():
            sums[key] = sums.get(key, 0) + value

    # Calculate averages
    averages = {key: sums[key] / len(metrics_dict) for key in sums}

    return averages


def create_evaluation_function(global_model, eval_datasets, tokenizer, global_metrics_history):

    def eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)

        print(f" {'-'*5} Global Eval in  Round {server_round} {'-'*5} ")

        all_metrics = {}
        total_loss = 0.0

        for dataset_name, dataset in eval_datasets.items():
            metrics = evaluate_llm(global_model, tokenizer, dataset)
            all_metrics[dataset_name] = metrics

            print(f"{dataset_name} || Metrics: {metrics}")

            total_loss += metrics['loss']

        _ = global_model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        avg_loss = total_loss / len(eval_datasets)

        print(f"Average Loss: {avg_loss:.2f}")

        metrics_record = {
            "round": server_round,
            "metrics_per_dataset": all_metrics,
            "avg_loss": avg_loss,
        }
        clients_states = CacheManager.get_clients_state()
        CacheManager.remove_clients_state()
        assert CacheManager.get_clients_state_count(
        ) == 0, "This should be zero after removing trained clients"

        if hasattr(global_model, 'peft_config'):
            model_state_dict = get_peft_model_state_dict(global_model)
            print("Using LoRA state dict for saving. For global model.")
        else:
            model_state_dict = global_model.state_dict()

        CacheManager.save_round_state(server_round, {
                                      "model": model_state_dict, "metrics": metrics_record}, clients_states)

        global_metrics_history.append(metrics_record)
        print(
            f"Clients state dict keys in round {server_round}: {list(clients_states.keys())}")

        print("=" * 60)
        return avg_loss, all_metrics

    return eval_gm


def create_server_fn(cfg, eval_datasets_dict, global_model, tokenizer,  global_metrics_history):
   

    # Ensure rounds state is cleared before starting
    CacheManager.remove_rounds_state()
    # Ensure clients state is cleared before starting
    CacheManager.remove_clients_state()
    init_ndarrays = ModelUtils.get_parameters(global_model)

    def server_fn(context):
        strategy = fl.server.strategy.FedAvg(
            min_evaluate_clients=0,
            fraction_evaluate=0,
            evaluate_fn=create_evaluation_function(
                global_model, eval_datasets_dict, tokenizer, global_metrics_history),
            initial_parameters=ndarrays_to_parameters(init_ndarrays),
        )
        server_config = fl.server.ServerConfig(
            num_rounds=cfg["fl"]["num_rounds"])
        return fl.server.ServerAppComponents(strategy=strategy, config=server_config)

    return server_fn
