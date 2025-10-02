import unsloth
import flwr as fl
import torch
import gc
from flwr.common import ndarrays_to_parameters

from src.utils.utils import CacheManager
from src.fl.util import ModelUtils, evaluate_llm
from src.utils.datasets import format_with_template


def create_evaluation_function(global_model, eval_datasets, tokenizer, global_metrics_history):

    def eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)

        print(
            f"========== GLOBAL MODEL EVALUATION - ROUND {server_round} ==========")

        all_metrics = {}
        total_loss = 0.0
        total_perplexity = 0.0
        dataset_count = 0

        for dataset_name, dataset in eval_datasets.items():
            dataset = format_with_template(tokenizer, dataset)
            metrics = evaluate_llm(global_model, tokenizer, dataset)
            all_metrics[dataset_name] = metrics

            print(
                f"{dataset_name.capitalize()} Dataset - Loss: {metrics['loss']:.2f}, Perplexity: {metrics['perplexity']:.2f}")

            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            dataset_count += 1

        _ = global_model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        avg_loss = total_loss / dataset_count
        avg_perplexity = total_perplexity / dataset_count

        print(
            f"Average          - Loss: {avg_loss:.2f}, Perplexity: {avg_perplexity:.2f}")

        metrics_record = {
            "round": server_round,
            "metrics_per_dataset": all_metrics,
            "avg_loss": avg_loss,
            "avg_perplexity": avg_perplexity
        }
        clients_states = CacheManager.get_clients_state()
        CacheManager.remove_clients_state()
        assert CacheManager.get_clients_state_count(
        ) == 0, "This should be zero after removing trained clients"
        CacheManager.save_round_state(server_round, {
                                      "model": global_model.state_dict(), "metrics": metrics_record}, clients_states)

        global_metrics_history.append(metrics_record)
        print(
            f"Clients state dict keys in round {server_round}: {list(clients_states.keys())}")
        
        print("=" * 60)
        return avg_loss, all_metrics

    return eval_gm


def create_server_fn(cfg, eval_datasets_dict, global_model, tokenizer,  global_metrics_history):
    import os
    import multiprocessing
    _original_cpu_count = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: 4

    if hasattr(os, 'cpu_count'):
        os.cpu_count = lambda: 4

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
