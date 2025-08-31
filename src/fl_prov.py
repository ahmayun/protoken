import torch
import logging
import torch.nn.functional as F


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
        logging.error(f"Inf values: {torch.sum(inf_mask)}")
        logging.error(f"NaN values: {torch.sum(nan_mask)}")
        logging.error(f"Total values: {torch.numel(t)}")
        # logging.error(f"Total values: {t}")
        raise ValueError("Anomalies detected in tensor")


def get_all_layers(net):
    all_layers = []
    for name, mode in net.named_modules():
        if name in ['lm_head', 'model.layers.17.mlp', 'model.layers.16.mlp',  'model.norm']:
            # print(f"Layer: {name}, appended")
            all_layers.append({"name": name, "layer": mode})
        # elif type(mode) == torch.nn.Linear:
        #     all_layers.append({"name": name, "layer": mode})
    return all_layers  # [len(all_layers)-2:]


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
        for h in self.all_hooks:
            h.remove()
        return {"activations": self.storage_forward, "gradients": self.storage_backward}


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
        for cli in client2layer_acts.keys():
            cli_acts = client2layer_acts[cli].flatten()
            _check_anomlies(cli_acts)
            cli_acts = cli_acts.to(dtype=gm_layer_grads.dtype)
            cli_part = torch.dot(cli_acts, gm_layer_grads)
            client2part[cli] = cli_part.item() * alpha_imp
        return client2part

    @staticmethod
    def _calculate_clients_contributions(gm_acts_grads_dict, client2layers, device):
        client2totalpart = {}

        layers_names = sorted(gm_acts_grads_dict["activations"].keys())
        for key in layers_names:
            
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

            c2contribution_per_layer = _normalize_with_softmax(c2contribution_per_layer)
            print(f"Layer: {key}, Contributions: {c2contribution_per_layer}")

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
        outputs = model(idx_cond)
        logits = outputs.logits[:, -1, :]  # last token prediction

        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        logits[0, next_token_id].backward()  # computing the gradients
        acts_grads_dict = hook_manager.get_hooks_data()

        return {"next_token_id": next_token_id, "acts_grads_dict": acts_grads_dict}

    @staticmethod
    def generate_text(model, client2model, tokenizer, prompt, terminators, max_new_tokens=64,
                      context_size=1024):
        model = model.cuda().eval()
        encoding = tokenizer(prompt, return_tensors="pt").to('cuda')
        idx = encoding["input_ids"]

        client2part = {cid: 0.0 for cid in client2model.keys()}
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            token_dict = ProvTextGenerator._get_next_token_id(model, idx_cond)
            next_token_id = token_dict["next_token_id"]
            temp_id = next_token_id.item()

            neuron_prov = NeuronProvenance(token_dict['acts_grads_dict'], client2model)
            conts_dict = neuron_prov.run()
            print(
                f"Token ID: {temp_id}, Decoded Token: {tokenizer.decode(temp_id)}, Contributions Dict: {conts_dict}")

            # mandatory to clear the gradients
            model.zero_grad(set_to_none=True)
            for c, v in conts_dict['client2part'].items():
                client2part[c] = client2part[c] + v

            if temp_id in terminators:
                # print(f" =====Found EOS token {temp_id}=====")
                break
            idx = torch.cat((idx, next_token_id), dim=-1)

        response = idx.squeeze(0)[encoding["input_ids"].shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=False)
        text = " ".join(text.split())
        # print(f'Response:\n ***|||{text}|||***\n\n')
        client2part =_normalize_with_softmax(client2part)
        # _ = input("Press Enter to continue")
        return {"response": text, "client2part": client2part}