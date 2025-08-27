
import time
from pathlib import Path


from diskcache import Index
import flwr as fl
import hydra
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from flwr.common import Context
from logging import DEBUG, INFO
import torch

from typing import Dict
from collections import OrderedDict


def config_sim_resources(cfg):
    """Configure the resources for the simulation."""
    client_resources = {"num_cpus": cfg.client_resources.num_cpus}
    if cfg.device == "cuda":
        client_resources["num_gpus"] = cfg.client_resources.num_gpus

    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}
    backend_config = {
        "client_resources": client_resources,
        "init_args": init_args,
        "working_dir": cfg.experiment_directories.root_dir
    }
    return backend_config


class ModelUtils:
    """Utility class for model parameter handling."""

    @staticmethod
    def _get_state_dict(net: torch.nn.Module) -> Dict:
        return net.state_dict()

    @staticmethod
    def get_parameters(model: torch.nn.Module) -> list:
        """Return model parameters as a list of NumPy ndarrays."""
        model = model.cpu()
        state_dict = ModelUtils._get_state_dict(model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: list) -> None:
        """Set model parameters from a list of NumPy ndarrays."""
        net = net.cpu()
        state_dict = ModelUtils._get_state_dict(net)
        params_dict = zip(state_dict.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        net.load_state_dict(state_dict, strict=True)


def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""
    log(INFO, ">>   ------------------- Clients Metrics ------------- ")
    all_logs = {}
    for nk_points, metric_d in metrics:
        cid = int(metric_d["cid"])
        temp_s = (
            f' Client {metric_d["cid"]}, Loss Train {metric_d["loss"]}, '
            f'Accuracy Train {metric_d["accuracy"]}, data_points = {nk_points}'
        )
        all_logs[cid] = temp_s

    # sorted by client id from lowest to highest
    for k in sorted(all_logs.keys()):
        log(INFO, all_logs[k])
    return {"loss": 0.0, "accuracy": 0.0}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

    def fit(self, parameters, config):
        trainer_args = config  # ["trainer_args"]
        print(f"client  {self.args['cid']}")
        client_train_response = self._train_fit(
            parameters, trainer_args)
        return client_train_response

    def _train_fit(self, parameters, trainer_args):
        model = self.args["model"]
        tokenizer = self.args["tokenizer"]

        ModelUtils.set_parameters(
            model, parameters=parameters)
        model = model.to(self.args["device"])
        train_dict = train_llm(
            model, tokenizer, self.args["client_data_train"])

        parameters = ModelUtils.get_parameters(model)

        client_train_dict = {"cid": self.args["cid"]} | train_dict
        nk_client_data_points = len(self.args["client_data_train"])

        return parameters, nk_client_data_points, client_train_dict


def fl_simulation(cfg):
    """Run the simulation."""
    global_model, tokenizer = get_model_and_tokenizer()

    fl_cache = Index(cfg.experiment_directories.fl_cache_dir)
    save_path = Path(HydraConfig.get().runtime.output_dir)
    # log(INFO, " ***********  Starting Experiment: %s ***************", exp_key)
    log(DEBUG, "Simulation Configuration: %s", cfg)

    round2gm_accs = []
    global global_round
    global_round = 0

    def _create_model():
        temp_model, _ = get_model_and_tokenizer()
        return temp_model

    def _get_fit_config(server_round):
        return cfg.hf_trainer_args

    def _eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)


        log(DEBUG, "config: %s", config)
        global global_round
        global_round += 1
        return -1, {"round": server_round}

    def _client_fn(context: Context):
        """Give the new client."""
        global global_round
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"

        args = {"cid": cid,
                "model": _create_model(),
                "tokenizer": tokenizer,
                "client_data_train": fl_dataset["client2dataset"][cid],
                'dir': save_path,
                }

        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        strategy = fl.server.strategy.FedAvg()
        server_config = fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds)
        return fl.server.ServerAppComponents(strategy=strategy, config=server_config)

    client_app = fl.client.ClientApp(client_fn=_client_fn)
    server_app = fl.server.ServerApp(server_fn=_server_fn)

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.fl.num_clients,
        backend_config=config_sim_resources(cfg),
    )


@hydra.main(config_path="./conf", config_name="casual_llm", version_base=None)
def main_fl(cfg) -> None:
    """Run the baseline."""
    start_time = time.time()
    fl_simulation(cfg)
    log(INFO, "Total Time Taken: %s seconds", time.time() - start_time)


if __name__ == "__main__":
    main_fl()
