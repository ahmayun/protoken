"""genfl: A Flower Baseline."""

from logging import INFO

import flwr as fl
from flwr.common.logger import log

from genfl.model import train, get_parameters, set_parameters



class FlowerClient(fl.client.NumPyClient):
    """Flower client for training a CNN model."""

    def __init__(self, args):
        """Initialize the client with the given configuration."""
        self.args = args

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        nk_client_data_points = len(self.args["client_data_train"])
        model = self.args["model"]

        set_parameters(model, parameters=parameters, peft=self.args["peft"])
        train_dict = train({"lr": config["lr"], "epochs": config["local_epochs"], "batch_size": config["batch_size"], "model": model,
                           "train_data": self.args["client_data_train"], "device": self.args["device"], "dir": self.args["dir"], })

        parameters = get_parameters(model, peft=self.args["peft"])

        client_train_dict = {"cid": self.args["cid"]} | train_dict

        log(INFO, "Client %s trained.", self.args["cid"])
        return parameters, nk_client_data_points, client_train_dict

