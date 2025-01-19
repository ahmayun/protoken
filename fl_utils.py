import random
import numpy as np
import torch


################
# Fixed Helper Functions
################


def set_exp_key(cfg):
    """Set the experiment key."""

    key = f"hello-genfl-1-"

    print(f"Default Key: {key}")

    # temp = input("Enter the experiment key to change: ")
    # if temp != "":
    #     key = temp
    # print(f"Experiment Key: {key}")

    return key


def config_sim_resources(cfg):
    """Configure the resources for the simulation."""
    client_resources = {"num_cpus": cfg.client_resources.num_cpus}
    if cfg.device == "cuda":
        client_resources["num_gpus"] = cfg.client_resources.num_gpus

    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": cfg.total_gpus}
    backend_config = {
        "client_resources": client_resources,
        "init_args": init_args,
        "working_dir":cfg.experiment_directories.root_dir
    }
    return backend_config


# def seed_everything(seed=786):
#     """Seed everything."""
