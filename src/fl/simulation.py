import flwr as fl
from src.utils.model import get_model_and_tokenizer
from src.fl.client import create_client_fn
from src.fl.server import create_server_fn
from src.utils.datasets import get_datasets_dict
import os
import multiprocessing
_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4


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


def run_fl_experiment(cfg):
    
    datasets_dict = get_datasets_dict(cfg["dataset"])
    global_metrics_history = []
    client_app = fl.client.ClientApp(client_fn=create_client_fn(cfg, datasets_dict['train']))

    global_model, tokenizer = get_model_and_tokenizer(cfg)
    server_app = fl.server.ServerApp(server_fn=create_server_fn(
        cfg, datasets_dict['test'], global_model, tokenizer, global_metrics_history))

    fl.simulation.run_simulation(server_app=server_app, client_app=client_app,
                                 num_supernodes=cfg["fl"]["num_clients"], backend_config=config_sim_resources(cfg))

    return global_metrics_history
