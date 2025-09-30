import unsloth
import flwr as fl
import torch
import math
import gc
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from flwr.common import ndarrays_to_parameters

from src.utils.utils import CacheManager
from src.utils.datasets import get_eval_datasets
from src.fl.util import ModelUtils, evaluate_llm



def create_evaluation_function(global_model, tokenizer, global_metrics_history, global_round_tracker):
    eval_datasets = get_eval_datasets(tokenizer)
    
    def eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)
        global_model_device = global_model

        print(
            f"========== GLOBAL MODEL EVALUATION - ROUND {server_round} ==========")

        all_metrics = {}
        total_loss = 0.0
        total_perplexity = 0.0
        dataset_count = 0

        for dataset_name, dataset in eval_datasets.items():
            metrics = evaluate_llm(
                global_model_device, tokenizer, dataset)
            all_metrics[dataset_name] = metrics

            print(
                f"{dataset_name.capitalize()} Dataset    - Loss: {metrics['loss']:.2f}, Perplexity: {metrics['perplexity']:.2f}")

            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            dataset_count += 1

        global_model_device = global_model_device.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        avg_loss = total_loss / dataset_count
        avg_perplexity = total_perplexity / dataset_count

        print(
            f"Average          - Loss: {avg_loss:.2f}, Perplexity: {avg_perplexity:.2f}")
        print("=" * 60)

        metrics_record = {
            "round": server_round,
            "chess_loss": all_metrics["chess"]["loss"],
            "chess_perplexity": all_metrics["chess"]["perplexity"],
            "math_loss": all_metrics["math"]["loss"],
            "math_perplexity": all_metrics["math"]["perplexity"],
            "avg_loss": avg_loss,
            "avg_perplexity": avg_perplexity
        }
        global_metrics_history.append(metrics_record)

        global_round_tracker[0] += 1

        CacheManager.save_temp_global_model(
            server_round, global_model.state_dict(), metrics_record)

        return avg_loss, all_metrics
    
    return eval_gm

def create_server_fn(global_model, tokenizer, cfg, global_metrics_history, global_round_tracker):
    def server_fn(context):
        init_ndarrays = ModelUtils.get_parameters(global_model)
        strategy = fl.server.strategy.FedAvg(
            min_evaluate_clients=0,
            fraction_evaluate=0,
            evaluate_fn=create_evaluation_function(global_model, tokenizer, global_metrics_history, global_round_tracker),
            initial_parameters=ndarrays_to_parameters(init_ndarrays),
        )
        server_config = fl.server.ServerConfig(
            num_rounds=cfg["fl"]["num_rounds"])
        return fl.server.ServerAppComponents(strategy=strategy, config=server_config)
    
    return server_fn
