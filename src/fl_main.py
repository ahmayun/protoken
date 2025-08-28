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
from typing import Dict, Any
from collections import OrderedDict

import unsloth
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

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

def dataset_adapter(name: str):
    name = name.lower()
    if name == "chess":
        hf_name = "Thytu/ChessInstruct"

        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": ex.get("task", "You are a helpful chess tutor.")},
                    {"role": "user", "content": ex.get("input")},
                    {"role": "assistant", "content": ex.get("expected_output")},
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

def build_trainer(model, tokenizer, train_dataset, cfg: Dict[str, Any]):
    args = cfg["train"]
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
            output_dir=str(args["output_dir"]),
            report_to=str(args["report_to"]),
        ),
    )

def get_client_dataset(cid: str):
    dataset_name = "chess" if cid == "0" else "math"
    hf_name, convert_to_chatml = dataset_adapter(dataset_name)
    
    split = "train[:10000]"
    dataset = load_dataset(hf_name, split=split)
    
    if dataset_name == "math":
        dataset = dataset.filter(lambda ex: ex.get("difficulty") == 'train-easy')
    
    dataset = dataset.map(convert_to_chatml)
    return dataset



def train_llm(model, tokenizer, dataset, cid):
    train_config = {
        "train": {
            "batch": 8,
            "ga": 1,
            "warmup_steps": 10,
            "max_steps": 20,
            "lr": 5e-5,
            "logging_steps": 5,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "scheduler": "linear",
            "seed": 3407,
            "output_dir": f"outputs_client_{cid}",
            "report_to": "none",
        }
    }
    
    formatted_dataset = format_with_template(tokenizer, dataset)
    trainer = build_trainer(model, tokenizer, formatted_dataset, train_config)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    
    print(f"> Starting training for client {cid}...")
    trainer.train()
    print(f"> Training completed for client {cid}.")
    
    return {"loss": -100.0, "accuracy": -100.0}

def config_sim_resources(cfg):
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

def _fit_metrics_aggregation_fn(metrics):
    log(INFO, ">>   ------------------- Clients Metrics ------------- ")
    all_logs = {}
    for nk_points, metric_d in metrics:
        cid = int(metric_d["cid"])
        temp_s = (
            f' Client {metric_d["cid"]}, Loss Train {metric_d["loss"]}, '
            f'Accuracy Train {metric_d["accuracy"]}, data_points = {nk_points}'
        )
        all_logs[cid] = temp_s

    for k in sorted(all_logs.keys()):
        log(INFO, all_logs[k])
    return {"loss": 0.0, "accuracy": 0.0}

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

def fl_simulation(cfg):
    global_model, tokenizer = get_model_and_tokenizer()

    save_path = Path(HydraConfig.get().runtime.output_dir)
    log(DEBUG, "Simulation Configuration: %s", cfg)

    global global_round
    global_round = 0

    def _create_model():
        temp_model, _ = get_model_and_tokenizer()
        return temp_model

    def _get_fit_config(server_round):
        return {}

    def _eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters)
        log(DEBUG, "config: %s", config)
        global global_round
        global_round += 1
        return -1, {"round": server_round}

    def _client_fn(context: Context):
        global global_round
        partition_id = context.node_config["partition-id"]
        cid = f"{partition_id}"

        args = {
            "cid": cid,
            "model": _create_model(),
            "tokenizer": tokenizer,
            "device": cfg.device,
            'dir': save_path,
        }

        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        strategy = fl.server.strategy.FedAvg(min_evaluate_clients=0, fraction_evaluate=0)
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
    start_time = time.time()
    fl_simulation(cfg)
    log(INFO, "Total Time Taken: %s seconds", time.time() - start_time)

if __name__ == "__main__":
    main_fl()
