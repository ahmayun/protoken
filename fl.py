# Standard Library Imports
import logging
import random
import time
from collections import Counter, OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Optional

# Third-Party Imports
import flwr as fl
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
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
from torch.utils.data import DataLoader
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



################
# Fixed Helper Functions
################

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



def get_model_and_tokenizer(model_cfg, peft_cfg, use_peft=True):

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)

    if peft_cfg.task_type == 'CAUSAL_LM':
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name, **model_cfg.kwargs)
    elif peft_cfg.task_type == 'SEQ_CLS':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name, **model_cfg.kwargs)

    if 'microsoft/Phi-3-mini-4k-instruct' not in model_cfg.name:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    if use_peft:
        peft_conf = LoraConfig(**peft_cfg)
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


def _cls_compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def _cls_hf_train_or_eval(model, hf_ds, train_cfg, do_train, do_eval):
    metric = evaluate.load("accuracy")
    training_args = TrainingArguments(
        do_train=do_train, do_eval=do_eval, **train_cfg)
    trainer = Trainer(model, training_args, train_dataset=hf_ds,
                      eval_dataset=hf_ds, compute_metrics=partial(_cls_compute_metrics, metric))

    if do_train:
        trainer.train()

    eval_result = trainer.evaluate()

    model = model.cpu()
    return {'accuracy': eval_result['eval_accuracy'], 'loss': eval_result['eval_loss']}


def load_datasets(dname):
    ds = load_dataset(dname, split="train")
    train_val, test = ds.train_test_split(test_size=0.10).values()
    train, val = train_val.train_test_split(test_size=0.2).values()

    # ratios of datasets pritn
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    print(f"Columns: {list(train.features)}")

    columns = list(train.features)

    return {"train": train, "val": val, "test": test, "columns": columns, 'ds': ds}


################
# Federated Learning
################


# fl utils
def get_labels_count(hf_dataset, target_label_col):
    label2count = Counter(example[target_label_col] for example in hf_dataset)
    return dict(label2count)


def get_correct_predictions_subset(model, test_data, hf_trainer_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    args = TrainingArguments(do_eval=True, do_train=False, **hf_trainer_config)
    trainer = Trainer(model=model, args=args,
                      eval_dataset=test_data, compute_metrics=None)

    # Use Trainer's predict method to get predictions
    predictions_output = trainer.predict(test_data)

    # Extract predictions and labels
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    # Create a boolean mask of correct predictions
    correct_mask = preds == labels
    correct_indices = np.where(correct_mask)[0].tolist()

    correct_subset = test_data.select(correct_indices)
    return correct_subset


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


def select_n_per_class(dataset, n=2):
    # Get all labels from the dataset
    labels = dataset['labels']
    unique_labels = set(labels)

    # Dictionary to store indices for each class
    class_indices = {label: [] for label in unique_labels}

    # Group indices by class
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Select n indices per class
    selected_indices = []
    for label in unique_labels:
        # Take first n indices for each class (or all if less than n available)
        indices = class_indices[label][:n]
        selected_indices.extend(indices)

    # Select the examples using the collected indices
    balanced_subset = dataset.select(selected_indices)
    return balanced_subset


def _run_provenance(gmodel, client2model, client2class, test_data, hf_trainer_config):
    # Get correct predictions subset

    correct_ds_subset = get_correct_predictions_subset(
        gmodel, test_data, hf_trainer_config)

    correct_ds_subset = select_n_per_class(correct_ds_subset, n=2)

    count = 0
    total_samples = len(correct_ds_subset)
    print(f"Total test data: {len(test_data)}")
    print(f"Correct subset size: {len(correct_ds_subset)}")
    print(f'Client2class: {client2class}')

    all_labels = set([l for label2count in client2class.values()
                     for l in label2count.keys()])

    # [c for c in client2class.keys() if label in client2class[c]]

    label2client = {label:  {c: client2class[c][label] for c in client2class.keys(
    ) if label in client2class[c]} for label in all_labels}
    print(f"Label2client: {label2client}")

    for i in range(total_samples):
        correct_subset = correct_ds_subset.select([i])

        label = correct_subset[0]['labels']

        if label not in label2client:
            continue

        true_responsible_clients = list(label2client[label].keys())
        res = provenance_of_fl_clients(
            gmodel=gmodel, c2model=client2model, test_data=correct_subset)

        client2part = {c: round(v, 3) for c, v in res['client2part'].items()}
        print(
            f"Label: {label} TClient: {res['traced_client']}, client2part: {client2part}, Label2client: {label2client[label]}")

        if res['traced_client'] in true_responsible_clients:
            count += 1

    accuracy = (count/total_samples) * 100
    print(
        f"Correctly traced clients: {count}/{total_samples}, Accuracy: {accuracy}%")
    # _ = input("Press any key to continue...")
    return accuracy


class ClientsAndServerDatasets:

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.client2dataset = {}
        self.server_dataset = None

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Initializing FederatedDataset.")
        # Initialize Flower FederatedDataset with the specified partitioner
        # IidPartitioner or DirichletPartitioner
        partitioner = self._initialize_partitioner()
        self.federated_dataset = FederatedDataset(
            dataset=cfg.dataset.name, partitioners={'train': partitioner})

        self.logger.info("Loading client partitions.")
        # Load client partitions
        self._load_client_partitions()

        if cfg.load_server_data == True:
            self.logger.info("Loading server data.")
            self._load_server_data()

        self.logger.info("DataLoaders prepared successfully.")

    def _initialize_partitioner(self):
        if self.cfg.dataset.distribution == "iid":
            self.logger.debug("Using IID partitioner.")
            return IidPartitioner(num_partitions=self.cfg.num_clients)
        elif self.cfg.dataset.distribution == "non_iid":
            self.logger.debug("Using Dirichlet partitioner.")
            return DirichletPartitioner(
                num_partitions=self.cfg.num_clients,
                alpha=self.cfg.dirichlet_alpha,
                min_partition_size=0,
                self_balancing=True,
            )

        elif self.cfg.dataset.distribution == "shard":
            self.logger.debug("Using Shard partitioner.")
            return PathologicalPartitioner(
                num_partitions=self.cfg.num_clients,
                partition_by=self.cfg.dataset.label_column,
                class_assignment_mode='deterministic',  # 'random',
                num_classes_per_partition=3,
            )

        else:
            raise ValueError(
                f"Unsupported distribution type: {self.cfg.distribution}")

    def _load_client_partitions(self):
        for cid in range(self.cfg.num_clients):
            client_id = f"{cid}"
            self.logger.debug(f"Loading partition for {client_id}.")
            partition = self.federated_dataset.load_partition(cid, 'train')
            tokenized_partition = self._tokenize_partition(partition)
            self.client2dataset[client_id] = tokenized_partition

    def _load_server_data(self):
        server_data = self.federated_dataset.load_split('test')
        tokenized_server = self._tokenize_partition(server_data)
        self.server_dataset = tokenized_server

    def _tokenize_partition(self, partition: Dataset) -> Dataset:
        self.logger.debug(
            "Applying tokenization transform to dataset partition.")

        text_column = self.cfg.dataset.text_column

        # Define the transformation function for a batch of examples
        def tokenize_batch(batch):
            # Tokenize the text data
            text = batch[text_column]
            tokenized_batch = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,  # This ensures that text longer than max_length is truncated
                max_length=self.cfg.dataset.max_length,
                # return_tensors="pt",
            )
            # Assign labels
            tokenized_batch["labels"] = batch[self.cfg.dataset.label_column]

            necessary_columns = ["input_ids", "attention_mask", "labels"]
            return {k: tokenized_batch[k] for k in necessary_columns}

        # Apply the transformation to the entire partition
        # on the fly
        partition = partition.select(range(2048)).map(
            tokenize_batch, batched=True)

        self.logger.debug("Tokenization transform applied successfully.")
        return partition

    def get_datasets(self) -> Dict[str, Optional[DataLoader]]:
        client2class = {c: get_labels_count(
            ds, 'labels') for c, ds in self.client2dataset.items()}

        return {
            "client2dataset": self.client2dataset,
            "server_dataset": self.server_dataset,
            'client2class': client2class
        }


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

    def fit(self, parameters, config):
        model = self.args["model"]

        ModelUtils.set_parameters(model, parameters=parameters, peft=self.args["peft"])
        train_dict = _cls_hf_train_or_eval(
            model, self.args["client_data_train"], config, do_train=True, do_eval=True)

        parameters = ModelUtils.get_parameters(model, peft=self.args["peft"])

        client_train_dict = {"cid": self.args["cid"]} | train_dict

        log(INFO, "Client %s trained.", self.args["cid"])
        nk_client_data_points = len(self.args["client_data_train"])
        return parameters, nk_client_data_points, client_train_dict


class FedAvgWithGenFL(fl.server.strategy.FedAvg):
    """FedAvg with Differential Testing."""

    def __init__(self, cfg, client2class, test_data, callback_create_model_fn, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.create_model_fn = callback_create_model_fn
        self.cfg = cfg
        self.test_data = test_data
        self.client2class = client2class

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients updates."""

        client2model = {fit_res.metrics["cid"]: self._to_pt_model(
            fit_res.parameters) for _, fit_res in results}

        # c2nk = {fit_res.metrics["cid"]: fit_res.metrics.get("data_points", 0) for _, fit_res in results}

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)
        # do provenance here

        global_model = self._to_pt_model(aggregated_parameters)
        res = _run_provenance(global_model, client2model,
                              self.client2class, self.test_data, self.cfg.hf_trainer_config)
        aggregated_metrics['provenance'] = res  # Add to metrics
        return aggregated_parameters, aggregated_metrics

    def _to_pt_model(self, parameters):
        """Convert parameters to state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model = self.create_model_fn()
        ModelUtils.set_parameters(model, ndarr, peft=self.cfg.peft)
        return model


################
# Provenance in FL
################


class HookManager:
    def __init__(self):
        self.storage = []
        self.all_hooks = []

    def insert_hook(self, layer, hook_type):
        def _forward_hook(module, input_tensor, output_tensor):
            input_tensor = input_tensor[0].detach()
            output_tensor = output_tensor.detach()
            self.storage.append((input_tensor, output_tensor))

        def _backward_hook(module, grad_input, grad_output):
            grad_input = grad_input[0].detach()
            grad_output = grad_output[0].detach()
            self.storage.append((grad_input, grad_output))

        if hook_type == 'forward':
            hook = layer.register_forward_hook(_forward_hook)
        elif hook_type == 'backward':
            hook = layer.register_full_backward_hook(_backward_hook)
        else:
            raise ValueError("Invalid hook type")
        self.all_hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.all_hooks:
            hook.remove()

    def get_hooks_data(self, hook_type):
        self._remove_hooks()
        temp_storage = self.storage
        if hook_type == 'backward':
            temp_storage.reverse()
            return temp_storage
        elif hook_type == 'forward':
            return temp_storage
        else:
            raise ValueError("Invalid hook type")


class NeuronProvenance:
    def __init__(self, gmodel, c2model, test_data):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.test_data = test_data
        self.gmodel = gmodel
        self.c2model = c2model
        self.client_ids = list(self.c2model.keys())
        # logging.info(f'client ids: {self.client_ids}')

    def _calculate_clients_contributions(self, gm_layers_ios, gm_layers_grads, client2layers):
        layers2prov = []
        for layer_id in range(len(gm_layers_ios)):
            c2l = {cid: client2layers[cid][layer_id]
                   for cid in self.client_ids}  # clients layer
            layer_inputs = gm_layers_ios[layer_id][0]  # layer inputs
            layer_grads = gm_layers_grads[layer_id][1]

            clinet2outputs = {c: _evaluate_layer(
                l, layer_inputs, device=self.device) for c, l in c2l.items()}
            c2contribution = _calculate_layer_contribution(
                gm_layer_grads=layer_grads, client2layer_acts=clinet2outputs)
            layers2prov.append(c2contribution)

        client2totalpart = {}
        for c2part in layers2prov:
            for cid in c2part.keys():
                client2totalpart[cid] = client2totalpart.get(
                    cid, 0) + c2part[cid]

        client2totalpart = _normalize_with_softmax(client2totalpart)
        return client2totalpart

    def run(self):
        data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1)
        batch_input = next(iter(data_loader))
        self.gmodel.eval()
        gm_layers_ios = _get_layers_io(self.gmodel, batch_input, self.device)
        gm_layers_grads = get_layers_gradients(
            self.gmodel, batch_input, self.device)
        client2layers = {cid: getAllLayers(cm)
                         for cid, cm in self.c2model.items()}

        client2part = self._calculate_clients_contributions(
            gm_layers_ios, gm_layers_grads, client2layers)
        traced_client = max(client2part, key=client2part.get)  # type: ignore
        return {"traced_client": traced_client, "client2part": client2part}


def _evaluate_layer(layer, input_tensor, device):
    layer.zero_grad()
    with torch.no_grad():
        layer = layer.eval().to(device)
        input_tensor = input_tensor.to(device)
        activations = layer(input_tensor).cpu()
        _ = layer.cpu()
        _ = input_tensor.cpu()
    return activations


def _calculate_layer_contribution(gm_layer_grads, client2layer_acts, alpha_imp=1):
    client2part = {cid: 0.0 for cid in client2layer_acts.keys()}
    # _checkAnomlies(global_neurons_outputs)
    _check_anomlies(gm_layer_grads)
    gm_layer_grads = gm_layer_grads.flatten().cpu()
    for cli in client2layer_acts.keys():
        cli_acts = client2layer_acts[cli].flatten().cpu()
        _check_anomlies(cli_acts)
        cli_acts = cli_acts.to(dtype=gm_layer_grads.dtype)
        cli_part = torch.dot(cli_acts, gm_layer_grads)
        client2part[cli] = cli_part.item() * alpha_imp
        _ = cli_acts.cpu()
    return client2part


def _normalize_with_softmax(contributions):
    conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
    client2prov = {cid: v.item()
                   for cid, v in zip(contributions.keys(), conts)}
    return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))


def _forward(net, text_input_tuple, device):
    net.to(device)
    # Assume text_input_tuple is already on the correct device and prepared
    text_input_tuple = {k: torch.tensor(v, device=device).unsqueeze(
        0) for k, v in text_input_tuple.items() if k in ["input_ids", "token_type_ids", "attention_mask"]}
    outs = net(**text_input_tuple)
    return outs


def _get_layers_io(model, test_data, device):
    hook_manager = HookManager()
    glayers = getAllLayers(model)
    _ = [hook_manager.insert_hook(layer, hook_type='forward')
         for layer in glayers]

    with torch.no_grad():
        _ = _forward(model, test_data, device)
    return hook_manager.get_hooks_data('forward')


def get_layers_gradients(net, text_input_tuple, device):
    # Insert hooks for capturing backward gradients of the transformer model
    hook_manager = HookManager()
    net.zero_grad()
    all_layers = getAllLayers(net)
    _ = [hook_manager.insert_hook(layer, hook_type='backward')
         for layer in all_layers]

    outs = _forward(net, text_input_tuple, device)
    logits = outs.logits  # Access the logits from the output object

    prob, predicted = torch.max(logits, dim=1)
    predicted = predicted.cpu().detach().item()
    logits[0, predicted].backward()  # computing the gradients

    gm_layers_grads = hook_manager.get_hooks_data('backward')
    return gm_layers_grads


def getAllLayers(net):
    layers = getAllLayersBert(net)
    return [layers[-1]]  # [len(layers)-1:len(layers)]


def getAllLayersBert(net):
    layers = []
    for layer in net.children():
        if isinstance(layer, (torch.nn.Linear)):
            layers.append(layer)
        elif len(list(layer.children())) > 0:
            temp_layers = getAllLayersBert(layer)
            layers = layers + temp_layers
    return layers


def _check_anomlies(t):
    inf_mask = torch.isinf(t)
    nan_mask = torch.isnan(t)
    if inf_mask.any() or nan_mask.any():
        logging.error(f"Inf values: {torch.sum(inf_mask)}")
        logging.error(f"NaN values: {torch.sum(nan_mask)}")
        logging.error(f"Total values: {torch.numel(t)}")
        # logging.error(f"Total values: {t}")
        raise ValueError("Anomalies detected in tensor")

# provenance of fl clients funtion


def provenance_of_fl_clients(gmodel, c2model, test_data):
    neuron_prov = NeuronProvenance(gmodel, c2model, test_data)
    return neuron_prov.run()


def run_simulation(cfg):
    """Run the simulation."""

    save_path = Path(HydraConfig.get().runtime.output_dir)

    exp_key = set_exp_key(cfg)

    log(INFO, " ***********  Starting Experiment: %s ***************", exp_key)

    log(DEBUG, "Simulation Configuration: %s", cfg)

    _, tokenizer = get_model_and_tokenizer(cfg.model_config, cfg.peft_config)

    ds_prep = ClientsAndServerDatasets(cfg, tokenizer)
    ds_dict = ds_prep.get_datasets()
    server_testdata = ds_dict["server_dataset"]

    round2gm_accs = []

    def _create_model():
        temp_model, _ = get_model_and_tokenizer(
            cfg.model_config, cfg.peft_config)
        return temp_model

    def _get_fit_config(server_round):
        return cfg.hf_trainer_config

    def _eval_gm(server_round, parameters, config):
        gm_model = _create_model()
        ModelUtils.set_parameters(gm_model, parameters, peft=cfg.peft)

        d_res = _cls_hf_train_or_eval(
            gm_model, server_testdata, cfg.hf_trainer_config, do_train=False, do_eval=True)

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
            'dir': save_path,
            'peft': cfg.peft
        }
        client = FlowerClient(args).to_client()
        return client

    def _server_fn(context: Context):
        initial_net = _create_model()
        strategy = FedAvgWithGenFL(
            cfg=cfg,  # Add cfg
            client2class=ds_dict["client2class"],
            test_data=server_testdata,  # Add test_data
            callback_create_model_fn=_create_model,
            accept_failures=False,
            fraction_fit=0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            initial_parameters=ndarrays_to_parameters(
                ndarrays=ModelUtils.get_parameters(initial_net, peft=cfg.peft)),  # Remove initial_parameters
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
        backend_config=config_sim_resources(cfg),
    )

    # utils.plot_metrics(round2gm_accs, round2feddebug_accs, cfg, save_path)

    log(INFO, "Training Complete for Experiment: %s", exp_key)


@hydra.main(config_path="./conf", config_name="cls_fl", version_base=None)
def main_fl(cfg) -> None:
    """Run the baseline."""
    start_time = time.time()
    run_simulation(cfg)
    log(INFO, "Total Time Taken: %s seconds", time.time() - start_time)


@hydra.main(config_path="./conf", config_name="central_ml", version_base=None)
def main_central_ml(cfg) -> None:
    ################
    # Model Loading
    ################
    model, tokenizer = get_model_and_tokenizer(
        cfg.model_config, cfg.peft_config)

    ds_dict = load_datasets(cfg.dataset_config.name)

    # Training
    train_conf = SFTConfig(**cfg.train_config)
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["val"],
        formatting_func=partial(
            create_alpaca_prompt_with_response, eos_token=tokenizer.eos_token),
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)

    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id, 50256]

    for e in ds_dict['ds']:
        prompt = _prompt(e['instruction'], e['input'])
        print(" ---------------- Start ----------------")
        print(f"Prompt: {prompt}")
        # generat_hf(model, tokenizer, terminators,  prompt)
        generate_self(model, tokenizer, terminators, prompt)
        print(" ---------------- End ----------------")
        _ = input("Press Enter to continue")


################
# Main
################

if __name__ == "__main__":
    main_fl()
    # main_central_ml()
