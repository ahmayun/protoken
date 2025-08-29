import time
from pathlib import Path
from diskcache import Index
import flwr as fl
from flwr.common.logger import log
from flwr.common import Context
from logging import DEBUG, INFO
import torch
from typing import Dict, Any
from collections import OrderedDict
import json

import unsloth
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

# choose a single GPU the server should use
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"   # or: os.environ["TORCHDYNAMO_DISABLE"] = "1"
from flwr.common import ndarrays_to_parameters

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

        parameters = ModelUtils.get_parameters(model)
        client_train_dict = {"cid": cid} | train_dict
        nk_client_data_points = len(dataset)

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


def save_and_plot_metrics(metrics_list, results_dir):

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "fl_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)

    if not metrics_list:
        print("No metrics to plot")
        return

    df = pd.DataFrame(metrics_list)

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(data=df, x='round', y='chess_loss',
                 label='Chess', ax=ax1, marker='o')
    sns.lineplot(data=df, x='round', y='math_loss',
                 label='Math', ax=ax1, marker='s')
    sns.lineplot(data=df, x='round', y='avg_loss',
                 label='Average', ax=ax1, marker='^')
    ax1.set_title('Loss vs Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    sns.lineplot(data=df, x='round', y='chess_perplexity',
                 label='Chess', ax=ax2, marker='o')
    sns.lineplot(data=df, x='round', y='math_perplexity',
                 label='Math', ax=ax2, marker='s')
    sns.lineplot(data=df, x='round', y='avg_perplexity',
                 label='Average', ax=ax2, marker='^')
    ax2.set_title('Perplexity vs Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = results_dir / "fl_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Metrics saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")


def get_config():
    return {
        "fl": {
            "num_rounds": 10,
            "num_clients": 2,
            "clients_per_round": 2
        },
        "train": {
            "batch": 8,
            "ga": 1,
            "warmup_steps": 10,
            "max_steps": 20,
            "lr": 5e-5,
            "logging_steps": 5,
            "optim": "adamw",
            "weight_decay": 0.01,
            "scheduler": "linear",
            "seed": 3407,
        },
        "model": {
            "name": "unsloth/gemma-3-270m-it",
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "load_in_8bit": False
        },
        "device": "cuda",
        "total_gpus": 1,
        "total_cpus": 16,
        "client_resources": {
            "num_cpus": 7,
            "num_gpus": 0.5
        }
    }


# Model, Training, Evaluations

def get_model_and_tokenizer():
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    return model, tokenizer


def _create_model():
    temp_model, _ = get_model_and_tokenizer()
    return temp_model


def build_trainer(model, tokenizer, train_dataset, args):

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=int(args["batch"]),
            gradient_accumulation_steps=int(args["ga"]),
            warmup_steps=int(args["warmup_steps"]),
            max_steps=int(args["max_steps"]),
            learning_rate=float(args["lr"]),
            logging_steps=int(args["logging_steps"]),
            optim=str(args["optim"]),
            weight_decay=float(args["weight_decay"]),
            lr_scheduler_type=str(args["scheduler"]),
            seed=int(args["seed"]),
            output_dir=None,
            report_to=None,
        ),
    )


def train_llm(model, tokenizer, dataset, cid):

    train_config = get_config()["train"]

    formatted_dataset = format_with_template(tokenizer, dataset)
    trainer = build_trainer(
        model, tokenizer, formatted_dataset, args=train_config)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print(f"> Starting training for client {cid}...")
    trainer.train()
    print(f"> Training completed for client {cid}.")

    return {"loss": -100.0, "accuracy": -100.0}



def evaluate_model_on_dataset(model, tokenizer, dataset, dataset_name):
    
    args = get_config()["train"]
    
    formatted_dataset = format_with_template(tokenizer, dataset)

    model = model.to("cuda:0")
    
    model.eval()
    trainer =  SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        eval_dataset=formatted_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_eval_batch_size=8,
            seed=int(args["seed"]),
            output_dir=None,
            report_to=None,
            do_train=False,
            do_eval=True,
            no_cuda=True, 
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    metrics =  trainer.evaluate()
    print(f"===========>>>>>>>>>>> WARIS Eval Metrics: {metrics}")

    return {"loss": metrics['eval_loss'] , "perplexity": math.exp(metrics['eval_loss'])}


# Dataset, Preprocessing, Chat Templates, etc
def dataset_adapter(name: str):
    name = name.lower()
    if name == "chess":
        hf_name = "Thytu/ChessInstruct"

        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": ex.get(
                        "task", "You are a helpful chess tutor.")},
                    {"role": "user", "content": ex.get("input")},
                    {"role": "assistant", "content": ex.get(
                        "expected_output")},
                ]
            }

        return hf_name, convert_to_chatml

    if name == "math":
        hf_name = "m-gopichand/deepmind_math_dataset_processed"

        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": "You are a helpful math tutor. Provide concise, correct solutions."},
                    {"role": "user", "content": ex.get("question")},
                    {"role": "assistant", "content": ex.get("answer")},
                ]
            }

        return hf_name, convert_to_chatml

    raise ValueError(f"Unknown dataset adapter: {name}")


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

    return dataset.map(formatting_prompts_func, batched=True)


def get_client_dataset(cid: str):
    dataset_name = "chess" if cid == "0" else "math"
    hf_name, convert_to_chatml = dataset_adapter(dataset_name)

    split = "train[:100]"
    dataset = load_dataset(hf_name, split=split)

    if dataset_name == "math":
        dataset = dataset.filter(
            lambda ex: ex.get("difficulty") == 'train-easy')

    dataset = dataset.map(convert_to_chatml)
    return dataset


def get_eval_datasets():
    # TODO: Use separate test datasets instead of train data
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

    def _eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)
        global_model_device = global_model.to(cfg["device"])

        eval_datasets = get_eval_datasets()

        print(
            f"========== GLOBAL MODEL EVALUATION - ROUND {server_round} ==========")

        all_metrics = {}
        total_loss = 0.0
        total_perplexity = 0.0
        dataset_count = 0

        for dataset_name, dataset in eval_datasets.items():
            metrics = evaluate_model_on_dataset(
                global_model_device, tokenizer, dataset, dataset_name)
            all_metrics[dataset_name] = metrics

            print(
                f"{dataset_name.capitalize()} Dataset    - Loss: {metrics['loss']:.2f}, Perplexity: {metrics['perplexity']:.2f}")

            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            dataset_count += 1

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

        return avg_loss, {"round": server_round, "avg_loss": avg_loss, "avg_perplexity": avg_perplexity} | all_metrics

    def _client_fn(context: Context):
        global global_round
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"

        args = {
            "cid": cid,
            "model": _create_model(),
            "tokenizer": tokenizer,
            "device": cfg["device"],
        }

        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        init_ndarrays = ModelUtils.get_parameters(global_model)  # from your already-loaded model
        strategy = fl.server.strategy.FedAvg(
            min_evaluate_clients=0,
            fraction_evaluate=0,
            evaluate_fn=_eval_gm,
            initial_parameters=ndarrays_to_parameters(init_ndarrays),  # 👈 seed server with real weights
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

    save_and_plot_metrics(global_metrics_history, results_dir)


def main():
    start_time = time.time()
    cfg = get_config()
    fl_simulation(cfg)
    print(f"Total Time Taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
