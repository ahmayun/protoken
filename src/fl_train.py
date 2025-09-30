import unsloth
import time
from pathlib import Path
import flwr as fl

import torch
from typing import Dict
from collections import OrderedDict

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import math
from flwr.common import ndarrays_to_parameters, Context
import logging
import gc
from src.plotting import save_and_plot_metrics
from src.utils import CacheManager


# Get the logger used by the Flower library
flwr_logger = logging.getLogger("flwr")

# Set its level to INFO to suppress DEBUG messages
flwr_logger.setLevel(logging.INFO)

# Stop the flwr logger from passing messages to the root logger
flwr_logger.propagate = False


import multiprocessing
import os
_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4



def get_config():
    return {
        "fl": {
            "num_rounds": 15,
            "num_clients": 2,
            "clients_per_round": 2
        },

        "sft_config_args": {
            # "dataset_text_field": "text",
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 15,
            "num_train_epochs": 1,
            "learning_rate": 5e-5,
            "logging_steps": 20,
            "optim": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "constant",
            "seed": 3407,
            "output_dir": None,
            "report_to": None,
        },

        "model_config": {
            "model_name": "unsloth/gemma-3-270m-it",
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "full_finetuning": True,
        },

        "chat_template": "gemma3",

        "device": "cuda",
        "total_gpus": 1,
        "total_cpus": 8,
        "client_resources": {
            "num_cpus": 6,
            "num_gpus": 1
        }
    }


class ModelUtils:
    @staticmethod
    def _get_state_dict(net: torch.nn.Module) -> Dict:
        return net.state_dict()

    @staticmethod
    def get_parameters(model: torch.nn.Module) -> list:
        model = model.cpu()
        state_dict = ModelUtils._get_state_dict(model)
        return [val.cpu().float().numpy() for _, val in state_dict.items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: list) -> None:
        net = net.cpu()
        state_dict = ModelUtils._get_state_dict(net)
        params_dict = zip(state_dict.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)


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

        # Only set parameters if they exist and are valid
        if parameters and len(parameters) > 0:
            print(f"Loading {len(parameters)} parameters for client {cid}")
            ModelUtils.set_parameters(model, parameters=parameters)
        else:
            print(f"No parameters to load for client {cid}, using fresh model")

        model = model.to(self.args["device"])
        dataset = get_client_dataset(cid)
        train_dict = train_llm(model, tokenizer, dataset, cid)
        model = model.to("cpu")

        # Save client model after training
        client_training_info = {
            "model_state_dict": model.state_dict(),
            "training_metrics": train_dict,
            "client_id": cid,
            "round": self.args['round'],
            "timestamp": time.time(),
            "dataset_size": len(dataset)
        }
        CacheManager.save_temp_client_model(
            cid, self.args['round'], client_training_info)

        parameters = ModelUtils.get_parameters(model)
        client_train_dict = {"cid": cid} | train_dict
        nk_client_data_points = len(dataset)

        del model
        del dataset
        torch.cuda.empty_cache()
        return parameters, nk_client_data_points, client_train_dict


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


# Model, Training, Evaluations

def get_model_and_tokenizer():
    model_config = get_config()["model_config"]
    get_chat_template_name = get_config()["chat_template"]
    model, tokenizer = FastModel.from_pretrained(**model_config)
    tokenizer = get_chat_template(
        tokenizer, chat_template=get_chat_template_name)
    return model, tokenizer


def _create_model():
    temp_model, _ = get_model_and_tokenizer()
    return temp_model


def train_llm(model, tokenizer, train_dataset, cid):

    sft_config_args = get_config()["sft_config_args"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        dataset_text_field="text",
        args=SFTConfig(**sft_config_args)
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_proc=int(8)
    )

    print(f"> Starting training for client {cid}...")
    metrics =  trainer.train().metrics
    print(f"> Training completed for client {cid} training loss: {metrics['train_loss']:.4f}")

    return {"loss": metrics['train_loss'], "perplexity": math.exp(metrics['train_loss'])}


def evaluate_model_on_dataset(model, tokenizer, eval_dataset):
    model.eval()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_eval_batch_size=32,
            seed=42,
            output_dir=None,
            report_to=None,
            do_train=False,
            do_eval=True,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_proc=8
    )

    metrics = trainer.evaluate()

    return {"loss": metrics['eval_loss'], "perplexity": math.exp(metrics['eval_loss'])}


def format_with_template(tokenizer, dataset):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True, num_proc=8)


def get_client_dataset(cid: str, num_samples=100):
    G_DS = load_dataset(
        'waris-gill/llm-datasets-instruct-for-FL', split="train")

    chess_ds = G_DS.filter(
        lambda label: label == "chess",
        input_columns="label",
        batched=False,
        num_proc=8,
    )

    math_ds = G_DS.filter(
        lambda label: label == "math",
        input_columns="label",
        batched=False,
        num_proc=8,
    )

    tokenizer = get_model_and_tokenizer()[1]

    dataset_map = {"0":  format_with_template(
        tokenizer, chess_ds), "1": format_with_template(tokenizer, math_ds)}
    return dataset_map.get(cid).select(range(num_samples))


def get_eval_datasets():
    chess_dataset = get_client_dataset("0")
    math_dataset = get_client_dataset("1")
    return {"chess": chess_dataset, "math": math_dataset}


def fl_simulation(cfg):
    global_model, tokenizer = get_model_and_tokenizer()

    print(f"Simulation Configuration: {cfg}")

    global global_round
    global_round = 0

    results_dir = Path("results")
    global_metrics_history = []

    eval_datasets = get_eval_datasets()

    def _eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)
        global_model_device = global_model

        print(
            f"========== GLOBAL MODEL EVALUATION - ROUND {server_round} ==========")

        all_metrics = {}
        total_loss = 0.0
        total_perplexity = 0.0
        dataset_count = 0

        for dataset_name, dataset in eval_datasets.items():
            metrics = evaluate_model_on_dataset(
                global_model_device, tokenizer, dataset)
            all_metrics[dataset_name] = metrics

            print(
                f"{dataset_name.capitalize()} Dataset    - Loss: {metrics['loss']:.2f}, Perplexity: {metrics['perplexity']:.2f}")

            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            dataset_count += 1

        global_model_device = global_model_device.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        avg_loss = total_loss / dataset_count
        avg_perplexity = total_perplexity / dataset_count

        print(
            f"Average          - Loss: {avg_loss:.2f}, Perplexity: {avg_perplexity:.2f}")
        print("=" * 60)

        metrics_record = {
            "round": server_round,
            "chess_loss": all_metrics["chess"]["loss"],
            "chess_perplexity": all_metrics["chess"]["perplexity"],
            "math_loss": all_metrics["math"]["loss"],
            "math_perplexity": all_metrics["math"]["perplexity"],
            "avg_loss": avg_loss,
            "avg_perplexity": avg_perplexity
        }
        global_metrics_history.append(metrics_record)

        global global_round
        global_round += 1

        CacheManager.save_temp_global_model(server_round, global_model.state_dict(), metrics_record)

        return avg_loss, all_metrics

    def _client_fn(context: Context):
        global global_round
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"

        args = {
            "cid": cid,
            "model": _create_model(),
            "tokenizer": tokenizer,
            "device": cfg["device"],
            'round': global_round
        }

        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        init_ndarrays = ModelUtils.get_parameters(
            global_model)  # from your already-loaded model
        strategy = fl.server.strategy.FedAvg(
            min_evaluate_clients=0,
            fraction_evaluate=0,
            evaluate_fn=_eval_gm,
            initial_parameters=ndarrays_to_parameters(
                init_ndarrays),  # 👈 seed server with real weights
        )
        server_config = fl.server.ServerConfig(
            num_rounds=cfg["fl"]["num_rounds"])
        return fl.server.ServerAppComponents(strategy=strategy, config=server_config)

    client_app = fl.client.ClientApp(client_fn=_client_fn)
    server_app = fl.server.ServerApp(server_fn=_server_fn)

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg["fl"]["num_clients"],
        backend_config=config_sim_resources(cfg),
    )

    return global_metrics_history, results_dir


def main():
    start_time = time.time()
    cfg = get_config()
    global_metrics_history, results_dir = fl_simulation(cfg)
    print(f"Total Time Taken: {time.time() - start_time} seconds")
    CacheManager.consolidate_experiment(
        exp_key="Test", num_rounds=cfg["fl"]["num_rounds"])
    save_and_plot_metrics(global_metrics_history, results_dir)


if __name__ == "__main__":
    main()
