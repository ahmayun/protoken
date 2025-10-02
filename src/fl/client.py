import flwr as fl
import torch
import time
import os
import warnings

from src.utils.utils import get_model_and_tokenizer
from src.fl.util import ModelUtils, train_llm
from src.utils.utils import CacheManager
from src.utils.datasets import format_with_template

# Avoid warnings
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
# warnings.filterwarnings("ignore", category=UserWarning)
import os
import multiprocessing
_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4
if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

    def fit(self, parameters, config):
        print(f"client  {self.args['cid']}")
        client_train_response = self._train_fit(parameters, config)
        return client_train_response

    def _train_fit(self, parameters, trainer_args):
        model = self.args["model"]
        tokenizer = self.args["tokenizer"]
        cid = self.args["cid"]
        dataset = format_with_template(tokenizer, self.args["dataset"])

        if parameters and len(parameters) > 0:
            print(f"Loading {len(parameters)} parameters for client {cid}")
            ModelUtils.set_parameters(model, parameters=parameters)
        else:
            print(f"No parameters to load for client {cid}, using fresh model")

        model = model.to(self.args["device"])
        train_dict = train_llm(model, tokenizer, dataset,
                               self.args["sft_config_args"])
        model = model.to("cpu")

        client_state = {
            "model": model.state_dict(),
            "metrics": train_dict,
            "client_id": cid,
            "timestamp": time.time(),
            "dataset_size": len(dataset),
        }
        CacheManager.save_client_trained_state(cid, client_state)


        parameters = ModelUtils.get_parameters(model)
        client_train_dict = {"cid": cid} | train_dict
        nk_client_data_points = len(dataset)

        del model
        del dataset
        torch.cuda.empty_cache()
        return parameters, nk_client_data_points, client_train_dict


def create_client_fn(cfg, train_dataset_dict):
    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"
        model, tokenizer = get_model_and_tokenizer(cfg)
        client_args = {
            "cid": cid,
            "model": model,
            "tokenizer": tokenizer,
            "device": cfg["device"],
            "dataset": train_dataset_dict[cid],
            "sft_config_args": cfg["sft_config_args"],
        }
        client = FlowerClient(client_args).to_client()
        return client
    return client_fn
