import flwr as fl
import torch
import time
import os
import warnings
from peft import get_peft_model_state_dict


from src.fl.model import ModelUtils, train_llm
from src.utils.cache import CacheManager


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

    def fit(self, parameters, config):
        print(f"client  {self.args['cid']}, client config, {config}")
        client_train_response = self._train_fit(parameters, config)
        return client_train_response

    def _train_fit(self, parameters, trainer_args):
        model = self.args["model"]
        tokenizer = self.args["tokenizer"]
        cid = self.args["cid"]
        # format_with_template(tokenizer, self.args["dataset"])
        dataset = self.args["dataset"]

        ModelUtils.set_parameters(model, parameters=parameters)
        model = model.to(self.args["device"])
        train_dict = train_llm(model, tokenizer, dataset,
                               self.args["sft_config_args"])
        model = model.to("cpu")

        if hasattr(model, 'peft_config'):
            model_state = get_peft_model_state_dict(model)
            print("Using LoRA state dict for saving. For client.")
        else:
            model_state = model.state_dict()

        client_state = {
            "model": model_state,
            "metrics": train_dict,
            "client_id": cid,
            "timestamp": time.time(),
            "dataset_size": len(dataset),
        }

        print(f'Storage-key: {self.args["storage_key"]}')

        CacheManager.save_client_trained_state(self.args["storage_key"], client_state)

        client_train_dict = {"cid": cid} | train_dict
        nk_client_data_points = len(dataset)

        trained_parameters = ModelUtils.get_parameters(model)
        del model
        del dataset
        torch.cuda.empty_cache()
        return trained_parameters, nk_client_data_points, client_train_dict
