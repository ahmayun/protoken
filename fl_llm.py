# Standard Library Imports
import logging
import math
import random
import time
from collections import Counter, OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Optional
import os
import warnings

# Third-Party Imports
import flwr as fl
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer, setup_chat_format
from hydra.core.hydra_config import HydraConfig
from flwr.common import ndarrays_to_parameters, Context
from logging import DEBUG, INFO
import evaluate

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


################
# Fixed Helper Functions
################
class ModelUtils:
    """Utility class for model parameter handling."""

    @staticmethod
    def _get_state_dict(net: torch.nn.Module, peft: bool) -> Dict:
        if peft:
            return get_peft_model_state_dict(net)
        else:
            return net.state_dict()

    @staticmethod
    def get_parameters(model: torch.nn.Module, peft: bool) -> list:
        """Return model parameters as a list of NumPy ndarrays."""
        model = model.cpu()
        state_dict = ModelUtils._get_state_dict(model, peft)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: list, peft: bool) -> None:
        """Set model parameters from a list of NumPy ndarrays."""
        net = net.cpu()
        state_dict = ModelUtils._get_state_dict(net, peft)
        params_dict = zip(state_dict.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        if peft:
            set_peft_model_state_dict(net, state_dict)
        else:
            net.load_state_dict(state_dict, strict=True)


def set_exp_key(cfg):
    """Set the experiment key."""
    key = f"hello fl world"
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
    }
    return backend_config


def seed_everything(seed=786):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prompt(instruction, input_of_instruction):
    prompt = f"### Instruction:\n{instruction}"
    if len(input_of_instruction) > 2:
        prompt += f"\n### Input:\n{input_of_instruction}"
    prompt += '\n### Response:\n'
    return prompt


def create_alpaca_prompt_with_response(example, eos_token):
    instruct = _prompt(example['instruction'], example['input'])
    prompt = instruct + example["output"] + " " + eos_token + " "
    return prompt


def get_model_and_tokenizer(model_cfg, peft):

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)

    if peft.config.task_type == 'CAUSAL_LM':
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name, **model_cfg.kwargs)
    elif peft.config.task_type == 'SEQ_CLS':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name, **model_cfg.kwargs)

    if 'microsoft/Phi-3-mini-4k-instruct' not in model_cfg.name:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    if peft.enabled:
        peft_conf = LoraConfig(**peft.config)
        model = get_peft_model(model, peft_conf)
        model.print_trainable_parameters()

    return model, tokenizer


def _manual_generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, terminators=None, tokenizer=None):
    model.eval()
    model = model.cuda()
    idx = idx.cuda()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            outputs = model(idx_cond)
            logits = outputs.logits

        logits = logits[:, -1, :]  # last token is the prediction

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # how does it get min val?
            logits = torch.where(
                logits < min_val, torch.tensor(float('-inf')), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        temp_id = next_token_id.item()

        if temp_id in terminators:
            print(terminators)
            print(
                f" =============================Found EOS token {temp_id} =============================")
            break
        idx = torch.cat((idx, next_token_id), dim=-1)
    return idx


def generate_self(model, tokenizer, terminators, prompt, max_new_tokens=256, context_size=1024):
    encoding = tokenizer(prompt, return_tensors="pt",).to('cuda')
    model = model.cuda().eval()
    with torch.no_grad():
        m_outs = _manual_generate(
            model, encoding["input_ids"], max_new_tokens=max_new_tokens, context_size=context_size,  terminators=terminators, tokenizer=tokenizer)
        m_outs = m_outs.squeeze(0)
        response = m_outs[encoding["input_ids"].shape[-1]:]
    text = tokenizer.decode(response, skip_special_tokens=False)
    text = " ".join(text.split())
    print(f'Response (Manual):\n ***|||{text}|||***\n\n')
    return text


def _casual_llm_hf_train_or_eval(model, tokenizer,  hf_ds, args_config, do_train, do_eval):
    train_conf = SFTConfig(do_train=do_train, do_eval=do_eval, **args_config)
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        train_dataset=hf_ds,
        eval_dataset=hf_ds,
        formatting_func=partial(
            create_alpaca_prompt_with_response, eos_token=tokenizer.eos_token),
    )
    metrics = {}
    if do_train:
        train_result = trainer.train()
        temp_metrics = train_result.metrics
        trainer.log_metrics("train", temp_metrics)
        metrics['loss'] = temp_metrics['train_loss']

    if do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        metrics['loss'] = eval_result['eval_loss']

    metrics['perplexity'] = math.exp(metrics['loss'])

    trainer.log_metrics("Generic Metrics", metrics)
    metrics['accuracy'] = -1
    return metrics


################
# Federated Learning
################


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


class ClientsAndServerDatasets:

    def __init__(self, cfg):
        self.cfg = cfg
        self.client2dataset = {}
        self.server_dataset = None
        partitioner = self._initialize_partitioner()
        self.federated_dataset = FederatedDataset(
            dataset=cfg.dataset.name, partitioners={'train': partitioner})

        for cid in range(self.cfg.fl.num_clients):
            client_id = f"{cid}"
            self.client2dataset[client_id] = self.federated_dataset.load_partition(
                cid, 'train').select(range(1*1024))

        if cfg.fl.load_server_data == True:
            self.server_dataset = self.federated_dataset.load_split(
                'train').select(range(1024))

    def _initialize_partitioner(self):
        if self.cfg.dataset.distribution == "iid":
            return IidPartitioner(num_partitions=self.cfg.fl.num_clients)
        elif self.cfg.dataset.distribution == "non_iid":
            return DirichletPartitioner(
                num_partitions=self.cfg.fl.num_clients,
                alpha=self.cfg.dirichlet_alpha,
                min_partition_size=0,
                self_balancing=True,
            )

        elif self.cfg.dataset.distribution == "shard":
            return PathologicalPartitioner(
                num_partitions=self.cfg.fl.num_clients,
                partition_by=self.cfg.dataset.label_column,
                class_assignment_mode='deterministic',  # 'random',
                num_classes_per_partition=3,
            )

        else:
            raise ValueError(
                f"Unsupported distribution type: {self.cfg.distribution}")

    def get_datasets(self):
        return {"client2dataset": self.client2dataset, "server_dataset": self.server_dataset}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

    def fit(self, parameters, config):
        model = self.args["model"]
        tokenizer = self.args["tokenizer"]

        ModelUtils.set_parameters(
            model, parameters=parameters, peft=self.args["peft"])
        train_dict = _casual_llm_hf_train_or_eval(
            model, tokenizer, self.args["client_data_train"], config, do_train=True, do_eval=True)

        parameters = ModelUtils.get_parameters(model, peft=self.args["peft"])

        client_train_dict = {"cid": self.args["cid"]} | train_dict

        log(INFO, "Client %s trained.", self.args["cid"])
        nk_client_data_points = len(self.args["client_data_train"])
        return parameters, nk_client_data_points, client_train_dict


class FedAvgWithGenFL(fl.server.strategy.FedAvg):
    """FedAvg with Differential Testing."""

    def __init__(self, callback_create_model_fn, callback_provenance_fn, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.callback_provenance_fn = callback_provenance_fn
        self.callback_create_model_fn = callback_create_model_fn

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients updates."""

        client2model = {fit_res.metrics["cid"]: self._to_pt_model(
            fit_res.parameters) for _, fit_res in results}

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)
        # do provenance here
        global_model = self._to_pt_model(aggregated_parameters)

        # self.callback_provenance_fn(global_model, client2model)

        return aggregated_parameters, aggregated_metrics

    def _to_pt_model(self, parameters):
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model = self.callback_create_model_fn()
        ModelUtils.set_parameters(model, ndarr, peft=True)
        return model


################
# Provenance in FL
################


class HookManager:
    def __init__(self):
        self.storage_forward = []
        self.storage_backward = []
        self.all_hooks = []

    def insert_hook(self, layer):
        def _forward_hook(module, input_tensor, output_tensor):
            input_tensor = input_tensor[0].detach()
            output_tensor = output_tensor.detach()
            self.storage_forward.append((input_tensor, output_tensor))

        def _backward_hook(module, grad_input, grad_output):
            grad_input = grad_input[0].detach()
            grad_output = grad_output[0].detach()
            self.storage_backward.append((grad_input, grad_output))

        hook_forward = layer.register_forward_hook(_forward_hook)
        hook_backward = layer.register_full_backward_hook(_backward_hook)
        self.all_hooks.append(hook_forward)
        self.all_hooks.append(hook_backward)

    def _remove_hooks(self):
        for hook in self.all_hooks:
            hook.remove()

    def get_hooks_data(self):
        self._remove_hooks()
        self.storage_backward.reverse()
        return {'activations': self.storage_forward, 'gradients': self.storage_backward}


class NeuronProvenance:
    def __init__(self,gm_acts_grads_dict, c2model):
        self.gm_acts_grads_dict = gm_acts_grads_dict
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.c2model = c2model
        self.client_ids = list(self.c2model.keys())

    @staticmethod
    def getAllLayers(net):
        layers = NeuronProvenance.getAllLayersBert(net)
        return [layers[-1]]  # [len(layers)-1:len(layers)]

    @staticmethod
    def getAllLayersBert(net):
        layers = []
        for layer in net.children():
            if isinstance(layer, (torch.nn.Linear)):
                layers.append(layer)
            elif len(list(layer.children())) > 0:
                temp_layers = NeuronProvenance.getAllLayersBert(layer)
                layers = layers + temp_layers
        return layers

    @staticmethod
    def _evaluate_layer(layer, input_tensor, device):
        layer.zero_grad()
        with torch.no_grad():
            layer = layer.eval().to(device)
            input_tensor = input_tensor.to(device)
            activations = layer(input_tensor).cpu()
            _ = layer.cpu()
            _ = input_tensor.cpu()
        return activations

    @staticmethod
    def _calculate_layer_contribution(gm_layer_grads, client2layer_acts, alpha_imp=1):
        client2part = {cid: 0.0 for cid in client2layer_acts.keys()}
        # _checkAnomlies(global_neurons_outputs)
        NeuronProvenance._check_anomlies(gm_layer_grads)
        gm_layer_grads = gm_layer_grads.flatten().cpu()
        for cli in client2layer_acts.keys():
            cli_acts = client2layer_acts[cli].flatten().cpu()
            NeuronProvenance._check_anomlies(cli_acts)
            cli_acts = cli_acts.to(dtype=gm_layer_grads.dtype)
            cli_part = torch.dot(cli_acts, gm_layer_grads)
            client2part[cli] = cli_part.item() * alpha_imp
            _ = cli_acts.cpu()
        return client2part

    @staticmethod
    def _normalize_with_softmax(contributions):
        conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
        client2prov = {cid: v.item()
                       for cid, v in zip(contributions.keys(), conts)}
        return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def _check_anomlies(t):
        inf_mask = torch.isinf(t)
        nan_mask = torch.isnan(t)
        if inf_mask.any() or nan_mask.any():
            logging.error(f"Inf values: {torch.sum(inf_mask)}")
            logging.error(f"NaN values: {torch.sum(nan_mask)}")
            logging.error(f"Total values: {torch.numel(t)}")
            # logging.error(f"Total values: {t}")
            raise ValueError("Anomalies detected in tensor")


    @staticmethod
    def _calculate_clients_contributions(gm_acts_grads_dict, client2layers, device):
        layers2prov = []
        for layer_id in range(len(gm_acts_grads_dict['activations'])):
            c2l = {cid: client2layers[cid][layer_id]
                   for cid in self.client_ids}  # clients layer
            # layer inputs
            layer_inputs = gm_acts_grads_dict['activations'][layer_id][0]
            layer_grads = gm_acts_grads_dict['gradients'][layer_id][1]

            clinet2outputs = {c: NeuronProvenance._evaluate_layer(l, layer_inputs, device) for c, l in c2l.items()}
            c2contribution = NeuronProvenance._calculate_layer_contribution(
                gm_layer_grads=layer_grads, client2layer_acts=clinet2outputs)
            layers2prov.append(c2contribution)

        client2totalpart = {}
        for c2part in layers2prov:
            for cid in c2part.keys():
                client2totalpart[cid] = client2totalpart.get(
                    cid, 0) + c2part[cid]

        client2totalpart = NeuronProvenance._normalize_with_softmax(
            client2totalpart)
        return client2totalpart

    def run(self):
        client2layers = {cid: NeuronProvenance.getAllLayers(cm)
                         for cid, cm in self.c2model.items()}

        client2part = self._calculate_clients_contributions(self.gm_acts_grads_dict, client2layers)
        
        traced_client = max(client2part, key=client2part.get)  # type: ignore
        return {"traced_client": traced_client, "client2part": client2part}



class ProvTextGenerator:
    @staticmethod
    def _insert_hooks(model):
        # Insert hooks for capturing backward gradients of the transformer model
        model.eval()
        hook_manager = HookManager()
        model.zero_grad()
        all_layers = NeuronProvenance.getAllLayers(model)
        _ = [hook_manager.insert_hook(layer) for layer in all_layers]
        return hook_manager
    
    @staticmethod
    def _get_next_token_id(model, idx_cond):
        hook_manager =  ProvTextGenerator._insert_hooks(model)        
        outputs = model(idx_cond)
        logits = outputs.logits[:, -1, :]  # last token prediction
        
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        # temp_prob, temp_pred = torch.max(logits, dim=1, keepdim=True)

        # print(f"\nPredicted token: {temp_pred.item()} with probability: {temp_prob.item()}, next token: {next_token_id.item()}, withe probability: {logits[0, next_token_id].item()}")

        logits[0, next_token_id].backward()  # computing the gradients
        acts_grads_dict  = hook_manager.get_hooks_data()        
        
        return {"next_token_id": next_token_id, "acts_grads_dict": acts_grads_dict}

    @staticmethod
    def generate_text_with_prov(model, client2model, tokenizer, prompt, terminators=None, max_new_tokens=256,
                                context_size=1024):
        """Combined text generation function with manual token generation and decoding"""
        model = model.cuda().eval()  # global model
        encoding = tokenizer(prompt, return_tensors="pt").to('cuda')
        idx = encoding["input_ids"]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            token_dict = ProvTextGenerator._get_next_token_id(model, idx_cond)
            next_token_id = token_dict["next_token_id"]
            
            temp_id = next_token_id.item()
            if temp_id in terminators:
                print(f" =====Found EOS token {temp_id}=====")
                break
            idx = torch.cat((idx, next_token_id), dim=-1)

        response = idx.squeeze(0)[encoding["input_ids"].shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=False)
        text = " ".join(text.split())
        print(f'Response:\n ***|||{text}|||***\n\n')
        return text


def provenance_of_fl_clients(gmodel, c2model, test_data):
    neuron_prov = NeuronProvenance(gmodel, c2model, test_data)
    return neuron_prov.run()


def run_simulation(cfg):
    """Run the simulation."""

    save_path = Path(HydraConfig.get().runtime.output_dir)

    exp_key = set_exp_key(cfg)

    log(INFO, " ***********  Starting Experiment: %s ***************", exp_key)

    log(DEBUG, "Simulation Configuration: %s", cfg)

    global_model, tokenizer = get_model_and_tokenizer(cfg.model, cfg.peft)

    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_datasets()
    server_testdata = ds_dict["server_dataset"]

    round2gm_accs = []

    def _create_model():
        temp_model, _ = get_model_and_tokenizer(cfg.model, cfg.peft)
        return temp_model

    def _get_fit_config(server_round):
        return cfg.hf_trainer_args

    def _eval_gm(server_round, parameters, config):
        ModelUtils.set_parameters(global_model, parameters, peft=cfg.peft)
        d_res = _casual_llm_hf_train_or_eval(
            global_model, tokenizer, server_testdata, cfg.hf_trainer_args, do_train=False, do_eval=True)
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
            "tokenizer": tokenizer,
            "client_data_train": ds_dict["client2dataset"][cid],
            "device": torch.device(cfg.device),
            'dir': save_path,
            'peft': cfg.peft.enabled,
        }
        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        strategy = FedAvgWithGenFL(
            callback_create_model_fn=_create_model,
            callback_provenance_fn=partial(
                provenance_of_fl_clients, test_data=server_testdata),
            fraction_fit=0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.fl.clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=cfg.fl.num_clients,
            initial_parameters=ndarrays_to_parameters(
                ndarrays=ModelUtils.get_parameters(global_model, peft=cfg.peft.enabled)),  # Remove initial_parameters
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

    return global_model, tokenizer, server_testdata


@hydra.main(config_path="./conf", config_name="casual_llm", version_base=None)
def main_fl(cfg) -> None:
    """Run the baseline."""
    start_time = time.time()
    gm_model, tokenizer,  hf_ds = run_simulation(cfg)
    log(INFO, "Total Time Taken: %s seconds", time.time() - start_time)

    print(" ---------------- Testing Start ----------------")

    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id, 50256]
    for e in hf_ds:
        prompt = _prompt(e['instruction'], e['input'])
        print(" ---------------- Start ----------------")
        print(f"Prompt: {prompt}")
        # generat_hf(model, tokenizer, terminators,  prompt)
        generate_self(gm_model, tokenizer, terminators, prompt)

        print(" ---------------- End ----------------")
        ProvTextGenerator.generate_text_with_prov(gm_model, None, tokenizer, prompt, terminators)
        _ = input("Press Enter to continue")


if __name__ == "__main__":
    main_fl()
