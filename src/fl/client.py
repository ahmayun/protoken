import flwr as fl
import torch
import time
import os
import warnings

from src.utils.utils import CacheManager, get_model_and_tokenizer
from src.utils.datasets import get_client_dataset
from src.fl.util import ModelUtils, train_llm

# Avoid warnings
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
# warnings.filterwarnings("ignore", category=UserWarning)



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
        dataset = self.args["dataset"]

        if parameters and len(parameters) > 0:
            print(f"Loading {len(parameters)} parameters for client {cid}")
            ModelUtils.set_parameters(model, parameters=parameters)
        else:
            print(f"No parameters to load for client {cid}, using fresh model")

        model = model.to(self.args["device"])
        train_dict = train_llm(model, tokenizer, dataset, self.args["sft_config_args"])
        model = model.to("cpu")

        client_training_info = {
            "model_state_dict": model.state_dict(),
            "training_metrics": train_dict,
            "client_id": cid,
            "round": self.args['round'],
            "timestamp": time.time(),
            "dataset_size": len(dataset)
        }
        CacheManager.save_temp_client_model(cid, self.args['round'], client_training_info)

        parameters = ModelUtils.get_parameters(model)
        client_train_dict = {"cid": cid} | train_dict
        nk_client_data_points = len(dataset)

        del model
        del dataset
        torch.cuda.empty_cache()
        return parameters, nk_client_data_points, client_train_dict


def create_client_fn(cfg, tokenizer, global_round):
    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"


        client_args = {
            "cid": cid,
            "model": get_model_and_tokenizer(cfg)[0],
            "tokenizer": tokenizer,
            "device": cfg["device"],
            'round': global_round,
            "dataset": get_client_dataset(cid, tokenizer, num_samples=100),
            "sft_config_args": cfg["sft_config_args"]
        }

        client = FlowerClient(client_args).to_client()
        return client

    return client_fn
