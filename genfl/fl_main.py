# Standard Library Imports
import time
from functools import partial
from pathlib import Path
import os
import warnings
import random
import numpy as np
import torch

# Third-Party Imports
from diskcache import Index
import flwr as fl
import hydra
import torch
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from flwr.common import ndarrays_to_parameters, Context
from logging import DEBUG, INFO

from genfl.fl_model import ModelUtils, get_model_and_tokenizer, train_or_eval_llm
from genfl.fl_dataset import get_labels_count, Federate_Dataset
from genfl.fl_prov import ProvTextGenerator
from genfl.fl_utils import set_exp_key, config_sim_resources


seed = 786
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)




# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


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
        self.cache = self.args['clients_cache']
        self.client_key = self.args['client_key']
        self.use_cache_client_model = args['use_cache_client_model']

    def fit(self, parameters, config):
        trainer_args = config  # ["trainer_args"]
        print(f"client  {self.args['cid']} -> key {self.client_key}")

        if self.use_cache_client_model == True and self.client_key in self.cache.keys():
            client_train_response = self.cache[self.client_key]
            print(
                f"[Cach Hit] [Client-{self.args['cid']}] using its own cached model. ***** Train Dict: {client_train_response[2]} *****")
            return client_train_response

        client_train_response = self._train_fit(
            parameters, trainer_args)
        self.cache[self.client_key] = client_train_response
        print(f"Client {self.args['cid']} is cached.")
        return client_train_response

    def _train_fit(self, parameters, trainer_args):
        model = self.args["model"]
        tokenizer = self.args["tokenizer"]

        ModelUtils.set_parameters(
            model, parameters=parameters, peft=self.args["peft"])
        model = model.to(self.args["device"])
        train_dict = train_or_eval_llm(
            model, tokenizer, self.args["client_data_train"], trainer_args, do_train=True, do_eval=True)

        parameters = ModelUtils.get_parameters(model, peft=self.args["peft"])

        client_train_dict = {"cid": self.args["cid"]} | train_dict
        nk_client_data_points = len(self.args["client_data_train"])

        return parameters, nk_client_data_points, client_train_dict


class FedAvgWithGenFL(fl.server.strategy.FedAvg):
    """FedAvg with Differential Testing."""

    def __init__(self, peft, callback_create_model_fn, callback_provenance_fn, *args, **kwargs):
        """Initialize."""
        random.seed(seed)   
        super().__init__(*args, **kwargs)
        self.callback_provenance_fn = callback_provenance_fn
        self.callback_create_model_fn = callback_create_model_fn
        self.peft = peft


    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients updates."""

        client2model = {fit_res.metrics["cid"]: self._to_pt_model(
            fit_res.parameters) for _, fit_res in results}

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)
        # do provenance here
        global_model = self._to_pt_model(aggregated_parameters)

        self.callback_provenance_fn(global_model, client2model)

        return aggregated_parameters, aggregated_metrics

    def _to_pt_model(self, parameters):
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model = self.callback_create_model_fn()
        ModelUtils.set_parameters(model, ndarr, self.peft)
        return model


def prepare_datasets(exp_key, fl_cache, cfg):

    if exp_key in fl_cache.keys() and cfg.fl.use_cache_client_model:
        log(INFO, "Loading from cache")
        ds_dict, client2class = fl_cache[exp_key]
        return ds_dict, client2class

    ds_prep = Federate_Dataset(cfg)
    ds_dict = ds_prep.get_datasets()
    client2class = {k: get_labels_count(
        v) for k, v in ds_dict["client2dataset"].items()}
    fl_cache[exp_key] = ds_dict, client2class
    return ds_dict, client2class


def fl_simulation(cfg):
    """Run the simulation."""
    global_model, tokenizer = get_model_and_tokenizer(cfg.model, cfg.peft)

    fl_cache = Index(cfg.experiment_directories.fl_cache_dir)
    save_path = Path(HydraConfig.get().runtime.output_dir)
    exp_key = set_exp_key(cfg)
    log(INFO, " ***********  Starting Experiment: %s ***************", exp_key)
    log(DEBUG, "Simulation Configuration: %s", cfg)

    fl_dataset, client2class = prepare_datasets(
        cfg.experiment_key, fl_cache, cfg)
    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id, 50256]
    callback_prov_fn = partial(ProvTextGenerator.generate_batch_text, client2class=client2class, tokenizer=tokenizer,
                               terminators=terminators, batach_examples=fl_dataset['server_dataset'].select(range(10)))

    round2gm_accs = []
    global global_round
    global_round = 0

    def _create_model():
        temp_model, _ = get_model_and_tokenizer(cfg.model, cfg.peft)
        return temp_model

    def _get_fit_config(server_round):
        return cfg.hf_trainer_args

    def _eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters, peft=cfg.peft)
            
        d_res = train_or_eval_llm(
            global_model.to(cfg.device), tokenizer, fl_dataset['server_dataset'], cfg.hf_trainer_args, do_train=False, do_eval=True)
        round2gm_accs.append(d_res["accuracy"])
        log(DEBUG, "config: %s", config)
        global global_round
        global_round += 1

        return d_res["loss"], {
            "accuracy": d_res["accuracy"],
            "loss": d_res["loss"],
            "round": server_round,
        }

    def _client_fn(context: Context):
        """Give the new client."""
        global global_round
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"

        args = {
            "cid": cid,
            "model": _create_model(),
            "tokenizer": tokenizer,
            "client_data_train": fl_dataset["client2dataset"][cid],
            "device": torch.device(cfg.device),
            'dir': save_path,
            'peft': cfg.peft,
            'clients_cache': fl_cache,
            'use_cache_client_model': cfg.fl.use_cache_client_model,
            'client_key': exp_key + f"[round-{global_round}]" + f"-[client_id-{cid}]"
        }
        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        strategy = FedAvgWithGenFL(
            peft=cfg.peft,
            callback_create_model_fn=_create_model,
            callback_provenance_fn=callback_prov_fn,
            fraction_fit=0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.fl.clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=cfg.fl.num_clients,
            initial_parameters=ndarrays_to_parameters(
                ndarrays=ModelUtils.get_parameters(global_model, peft=cfg.peft)),  # Remove initial_parameters
            evaluate_fn=_eval_gm,
            on_fit_config_fn=_get_fit_config,
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
        )
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
    log(INFO, "Training Complete for Experiment: %s", exp_key)


@hydra.main(config_path="./conf", config_name="casual_llm", version_base=None)
def main_fl(cfg) -> None:
    """Run the baseline."""
    start_time = time.time()
    fl_simulation(cfg)
    log(INFO, "Total Time Taken: %s seconds", time.time() - start_time)


if __name__ == "__main__":
    main_fl()
