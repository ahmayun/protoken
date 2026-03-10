import flwr as fl
from peft import get_peft_model_state_dict
import torch
import gc
from flwr.common import ndarrays_to_parameters

from src.fl.model import get_model_and_tokenizer
from src.dataset.datasets import get_datasets_dict
from src.utils.cache import CacheManager
from src.fl.model import ModelUtils, evaluate_llm
from src.fl.client import FlowerClient
GLOBAL_ROUND = 0

class MySampler(fl.server.SimpleClientManager):
    def sample( self, num_clients: int, min_num_clients = None, criterion = None):
        global GLOBAL_ROUND
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        import random 
        random.seed(GLOBAL_ROUND)
        
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            print(
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        # for k,v in self.clients.items():
        #     print(f"Available client id: {k}")
        #     print(f"Client info: {v.cid}, {v.node_id}, {v.properties}")
        

        sampled_cids = random.sample(available_cids, num_clients)
        partition_based_ids = [self.clients[cid] for cid in sampled_cids]
        # print(f"\n\n >> Sampled clients IDS: {partition_based_ids} << \n\n")
        return partition_based_ids


def config_sim_resources(cfg):
    client_resources = {"num_cpus": cfg["client_resources"]["num_cpus"]}
    if cfg["device"] == "cuda":
        client_resources["num_gpus"] = cfg["client_resources"]["num_gpus"]

    init_args = {"num_cpus": cfg["total_cpus"], "num_gpus": cfg["total_gpus"]}
    backend_config = {
        "client_resources": client_resources,
        "init_args": init_args,
        "working_dir": "outputs"
    }
    return backend_config


def average_dicts(metrics_dict):
    sums = {}
    for d in metrics_dict.values():
        for key, value in d.items():
            sums[key] = sums.get(key, 0) + value

    # Calculate averages
    averages = {key: sums[key] / len(metrics_dict) for key in sums}

    return averages


def create_evaluation_function(exp_key, global_model, eval_datasets, tokenizer, global_metrics_history, device="cuda"):

    def eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)

        print(f" {'-'*5} Global Eval in  Round {server_round} {'-'*5} ")

        all_metrics = {}
        total_loss = 0.0

        for dataset_name, dataset in eval_datasets.items():
            global_model.to(device)
            metrics = evaluate_llm(global_model, tokenizer, dataset)
            all_metrics[dataset_name] = metrics

            print(f"{dataset_name} || Metrics: {metrics}")

            total_loss += metrics['loss']

            # Free GPU after each eval dataset to avoid OOM with multiple datasets
            _ = global_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
        avg_loss = total_loss / len(eval_datasets)

        metrics_record = {
            "round": server_round,
            "metrics_per_dataset": all_metrics,
            "avg_loss": avg_loss,
        }
   

        if hasattr(global_model, 'peft_config'):
            model_state_dict = get_peft_model_state_dict(global_model)
            print("Using LoRA state dict for saving. For global model.")
        else:
            model_state_dict = global_model.state_dict()

        

        CacheManager.save_global_state(exp_key,  server_round, {
                                      "model": model_state_dict, "metrics": metrics_record})

        global_metrics_history.append(metrics_record)
        

        model_state_dict.clear()
        gc.collect()
        print(f"\n\n>> Average Loss: {avg_loss:.2f} in Round {server_round}")
        print("=" * 60)

        global GLOBAL_ROUND
        GLOBAL_ROUND += 1
        
        return avg_loss, all_metrics

    return eval_gm


def create_server_fn(exp_key, cfg, eval_datasets_dict, global_model, tokenizer,  global_metrics_history):
    init_ndarrays = ModelUtils.get_parameters(global_model)

    def server_fn(context):
        strategy = fl.server.strategy.FedAvg(
            min_evaluate_clients=0,
            fraction_evaluate=0,
            fraction_fit=0,
            evaluate_fn=create_evaluation_function(exp_key,
                                                   global_model, eval_datasets_dict, tokenizer, global_metrics_history,
                                                   device=cfg["device"]),
            initial_parameters=ndarrays_to_parameters(init_ndarrays),
            min_fit_clients=cfg["fl"]["clients_per_round"],
            accept_failures=False,
            min_available_clients=cfg["fl"]["num_clients"],
        )
        server_config = fl.server.ServerConfig(
            num_rounds=cfg["fl"]["num_rounds"])
        return fl.server.ServerAppComponents(strategy=strategy, config=server_config, client_manager=MySampler())
    return server_fn


def create_client_fn(exp_key, cfg, train_dataset_dict):
    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"
        model, tokenizer = get_model_and_tokenizer(cfg)
        global GLOBAL_ROUND
        client_args = {
            "cid": cid,
            "model": model,
            "tokenizer": tokenizer,
            "device": cfg["device"],
            "dataset": train_dataset_dict[cid],
            "sft_config_args": cfg["sft_config_args"],
            "use_lora": cfg.get("lora_config", {}).get("use_lora", False),
            "storage_key": f"{exp_key}-round-{GLOBAL_ROUND}-client-{cid}" 
        }
        client = FlowerClient(client_args).to_client()
        return client
    return client_fn


def run_fl_experiment(exp_key, cfg):
    CacheManager.clear_training_with_key(exp_key)

    # Flower simulation uses Ray as its backend. Ray is not available on all
    # Python versions/distributions, so fail early with a helpful message.
    try:
        import ray  # noqa: F401
    except Exception as e:
        import sys
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise ImportError(
            "Flower simulation requires the `ray` backend, but `ray` could not be imported.\n"
            f"Detected Python {pyver}. Use Python 3.12 and install with `flwr[simulation]` "
            "(or run `./setup.sh` which pins a Python 3.12 env)."
        ) from e

    datasets_dict = get_datasets_dict(
        cfg["fl"]["num_clients"],  **cfg["dataset"])
    global_metrics_history = []
    client_app = fl.client.ClientApp(
        client_fn=create_client_fn(exp_key, cfg, datasets_dict['train']))

    global_model, tokenizer = get_model_and_tokenizer(cfg)
    server_app = fl.server.ServerApp(server_fn=create_server_fn(exp_key,
                                                                cfg, datasets_dict['test'], global_model, tokenizer, global_metrics_history))

    fl.simulation.run_simulation(server_app=server_app, client_app=client_app,
                                 num_supernodes=cfg["fl"]["num_clients"], backend_config=config_sim_resources(cfg))

    global GLOBAL_ROUND
    GLOBAL_ROUND = 0
    return global_metrics_history
