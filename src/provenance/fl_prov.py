import torch
import logging
import torch.nn.functional as F
import torch.nn as nn

# Use a module-specific logger named "prov"
logger = logging.getLogger("prov")
logger.setLevel(logging.INFO)  # or logging.INFO as needed
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    logger.addHandler(handler)



def _get_lora_layers(net):
    all_layers = []
    for name, layer in net.named_modules():
        if name.find('lora_dropout') == -1 and name.endswith('.mlp.down_proj.lora_A.default'):
            all_layers.append({"name": name, "layer": layer})

    layers_to_return = all_layers[-2:]
    return layers_to_return


def get_all_layers(net):
    if hasattr(net, 'peft_config'):
        return _get_lora_layers(net)

    all_layers = []
    for name, mode in net.named_modules():
         for layer_name in ['lm_head', '.mlp']:
            if name.endswith(layer_name):
                all_layers.append({"name": name, "layer": mode})
    return all_layers[-4:]


def _insert_hooks_and_get_hooks_manger(model):
    model.eval()
    model.zero_grad()
    hook_manager = HookManager()
    layers = get_all_layers(model)

    for layer_dict in layers:
        hook_manager.insert_hook(
            layer_dict['layer'], key=layer_dict['name'])  # <-- stable key

    return hook_manager


def _check_anomlies(t):
    inf_mask = torch.isinf(t)
    nan_mask = torch.isnan(t)
    if inf_mask.any() or nan_mask.any():
        logger.error(f"Inf values: {torch.sum(inf_mask)}")
        logger.error(f"NaN values: {torch.sum(nan_mask)}")
        logger.error(f"Total values: {torch.numel(t)}")
        # logger.error(f"Total values: {t}")
        raise ValueError("Anomalies detected in tensor")





def _normalize_with_softmax(contributions):
    conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
    client2prov = {cid: v.item()
                   for cid, v in zip(contributions.keys(), conts)}
    return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))


class HookManager:
    def __init__(self):
        self.storage_forward = {}
        self.storage_backward = {}
        self.all_hooks = []

    def insert_hook(self, layer, key):
        def _forward_hook(module, input_tensor, output_tensor):
            x = input_tensor[0].detach()
            # y = output_tensor.detach()
            # self.storage_forward[key] = (x, y)
            self.storage_forward[key] = x

        def _backward_hook(module, grad_input, grad_output):
            # gx = grad_input[0].detach()
            gy = grad_output[0].detach()
            # self.storage_backward[key] = (gx, gy)
            self.storage_backward[key] = gy

        self.all_hooks += [
            layer.register_forward_hook(_forward_hook),
            layer.register_full_backward_hook(_backward_hook),
        ]

    def get_hooks_data(self):
        result = {"activations": self.storage_forward,
                  "gradients": self.storage_backward}
        for h in self.all_hooks:
            h.remove()
        return result


class NeuronProvenance:
    def __init__(self, gm_acts_grads_dict, c2model):
        self.gm_acts_grads_dict = gm_acts_grads_dict
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.c2model = c2model

    @staticmethod
    def _evaluate_layer(layer, input_tensor):
        with torch.no_grad():
            layer = layer.eval()
            activations = layer(input_tensor)
        return activations

    @staticmethod
    def _calculate_layer_contribution(gm_layer_grads, client2layer_acts, alpha_imp=1):
        client2part = {cid: 0.0 for cid in client2layer_acts.keys()}
        _check_anomlies(gm_layer_grads)
        gm_layer_grads = gm_layer_grads.flatten()
        for cli in client2part.keys():
            cli_acts = client2layer_acts[cli].flatten()
            _check_anomlies(cli_acts)
            cli_acts = cli_acts.to(dtype=gm_layer_grads.dtype)
            cli_part = torch.dot(cli_acts, gm_layer_grads)
            client2part[cli] = cli_part.item() * alpha_imp
        return client2part

    @staticmethod
    def _calculate_clients_contributions(gm_acts_grads_dict, client2layers, device):
        client2totalpart = {}

        all_layers_names = list(gm_acts_grads_dict["activations"].keys())

        # temp = gm_acts_grads_dict["gradients"].keys()
        # for name in all_layers_names:
        #     print (f"{name}, has grads: {name in temp}")

        # _ = input("Press Enter to continue...")

        for key in all_layers_names:
            # print(key)

            layer_inputs = gm_acts_grads_dict["activations"][key]
            layer_grads = gm_acts_grads_dict["gradients"][key]
            c2l = {}
            for cid, client_layers in client2layers.items():
                for layer_dict in client_layers:
                    if layer_dict['name'] == key:
                        c2l[cid] = layer_dict['layer']
                        break

            # dtype/device alignment
            c2acts = {cid: NeuronProvenance._evaluate_layer(
                l, layer_inputs) for cid, l in c2l.items()}

            c2contribution_per_layer = NeuronProvenance._calculate_layer_contribution(
                gm_layer_grads=layer_grads, client2layer_acts=c2acts)

            # c2contribution_per_layer = _normalize_with_softmax(c2contribution_per_layer)
            # logger.debug(f"Layer: {key}, Contributions: {c2contribution_per_layer}")
            logger.debug(f"Layer: {key}, Contributions: {c2contribution_per_layer}")

            for cid, v in c2contribution_per_layer.items():
                client2totalpart[cid] = client2totalpart.get(cid, 0.0) + v

        client2totalpart = _normalize_with_softmax(client2totalpart)
        return client2totalpart

    def run(self):
        client2layers = {cid: get_all_layers(cm)
                         for cid, cm in self.c2model.items()}

        client2part = self._calculate_clients_contributions(
            self.gm_acts_grads_dict, client2layers, self.device)

        traced_client = max(client2part, key=client2part.get)  # type: ignore
        return {"traced_client": traced_client, "client2part": client2part}


class ProvTextGenerator:

    @staticmethod
    def _get_next_token_id(model, idx_cond):
        hook_manager = _insert_hooks_and_get_hooks_manger(model)
        model.zero_grad(set_to_none=True)
        model.eval()
        model.train()  # to enable gradients

        # 1. Perform a forward pass to get the logits for the next token
        outputs = model(idx_cond)
        logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        logits[0, next_token_id].backward()  # computing the gradients

        # obj = logits.log_softmax(dim=-1).gather(1, next_token_id).sum()
        # obj.backward()

        acts_grads_dict = hook_manager.get_hooks_data()

        return {"next_token_id": next_token_id, "acts_grads_dict": acts_grads_dict}

    @staticmethod
    def generate_text(model, client2model, tokenizer, prompt, max_new_tokens=64,
                      context_size=2048):
        terminal_ids = [tokenizer.eos_token_id]

        try:
            end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            terminal_ids.append(end_of_turn_id)
        except:
            logger.warning("double check if <end_of_turn> token exists in the tokenizer")

        encoding = tokenizer(prompt, return_tensors="pt").to('cuda')
        idx = encoding["input_ids"]

        client2part = {cid: 0.0 for cid in client2model.keys()}
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            token_dict = ProvTextGenerator._get_next_token_id(model, idx_cond)
            next_token_id = token_dict["next_token_id"]
            temp_id = next_token_id.item()

            neuron_prov = NeuronProvenance(
                token_dict['acts_grads_dict'], client2model)
            conts_dict = neuron_prov.run()

            logger.debug(
                f"Token ID: {temp_id}, Decoded Token: {tokenizer.decode(temp_id)}, Contributions Dict: {conts_dict}")

            # mandatory to clear the gradients
            model.zero_grad(set_to_none=True)
            for c, v in conts_dict['client2part'].items():
                client2part[c] = client2part[c] + v

            if temp_id in terminal_ids:
                break

            idx = torch.cat((idx, next_token_id), dim=-1)

        response = idx.squeeze(0)[encoding["input_ids"].shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=False)
        text = " ".join(text.split())
        client2part = _normalize_with_softmax(client2part)
        return {"response": text, "client2part": client2part}
