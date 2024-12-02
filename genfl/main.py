import time
from logging import DEBUG, INFO
from pathlib import Path

import flwr as fl
import hydra
import torch
from flwr.common import ndarrays_to_parameters, Context
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig

# from flwr.simulation import run_simulation


from genfl import utils
from genfl.client import FlowerClient
from genfl.dataset import ClientsAndServerDatasets
from genfl.model import initialize_model, test
from genfl.strategy import FedAvgWithGenFL

utils.seed_everything(786)


def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""
    log(INFO, ">>   ------------------- Clients Metrics ------------- ")
    all_logs = {}
    for nk_points, metric_d in metrics:
        cid = int(metric_d["cid"])
        temp_s = (
            f' Client {metric_d["cid"]}, Loss Train {metric_d["train_loss"]}, '
            f'Accuracy Train {metric_d["train_accuracy"]}, data_points = {nk_points}'
        )
        all_logs[cid] = temp_s

    # sorted by client id from lowest to highest
    for k in sorted(all_logs.keys()):
        log(INFO, all_logs[k])
    return {"loss": 0.0, "accuracy": 0.0}


def run_simulation(cfg):
    """Run the simulation."""

    save_path = Path(HydraConfig.get().runtime.output_dir)

    exp_key = utils.set_exp_key(cfg)

    log(INFO, " ***********  Starting Experiment: %s ***************", exp_key)

    log(DEBUG, "Simulation Configuration: %s", cfg)

    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_datasets()
    server_testdata = ds_dict["server_dataset"]

    round2gm_accs = []
    

    def _create_model():
        return initialize_model(cfg.model, cfg.dataset.num_classes)

    def _get_fit_config(server_round):
        return {
            "server_round": server_round,
            "local_epochs": cfg.client.epochs,
            "batch_size": cfg.client.batch_size,
            "lr": cfg.client.lr,
        }

    def _eval_gm(server_round, parameters, config):
        gm_model = _create_model()
        utils.set_parameters(gm_model, parameters)
        test_cfg = {'model': gm_model, 'test_data':server_testdata, 'dir': save_path}
        d_res = test(test_cfg)
        round2gm_accs.append(d_res["accuracy"])
        log(DEBUG, "config: %s", config)
        return d_res["loss"], {
            "accuracy": d_res["accuracy"],
            "loss": d_res["loss"],
            "round": server_round,
        }
    
    def _client_fn(context: Context):
        """Give the new client."""
        # print("context", context)
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}" 

        args = {
            "cid": cid,
            "model": _create_model(),
            "client_data_train": ds_dict["client2dataset"][cid],
            "device": torch.device(cfg.device),
            'dir': save_path
        }
        client = FlowerClient(args).to_client()
        return client
    


    def _server_fn(context: Context):
        initial_net = _create_model()
        strategy = FedAvgWithGenFL(       
            device=cfg.device,
            callback_create_model_fn=_create_model,
            accept_failures=False,
            fraction_fit=0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            initial_parameters=ndarrays_to_parameters(ndarrays=utils.get_parameters(initial_net)),
            evaluate_fn=_eval_gm,
            on_fit_config_fn=_get_fit_config,
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
        )
        server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)

        return fl.server.ServerAppComponents(strategy=strategy, config=server_config)

        
    
    client_app = fl.client.ClientApp(client_fn=_client_fn)
    server_app = fl.server.ServerApp(server_fn=_server_fn)
    

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=utils.config_sim_resources(cfg),
    )

    # utils.plot_metrics(round2gm_accs, round2feddebug_accs, cfg, save_path)

    log(INFO, "Training Complete for Experiment: %s", exp_key)


@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg) -> None:
    """Run the baseline."""
    start_time = time.time()
    run_simulation(cfg)
    log(INFO, "Total Time Taken: %s seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
