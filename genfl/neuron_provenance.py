import logging
import torch
import torch.nn.functional as F
from genfl.model import test


def _check_anomlies(t):
    inf_mask = torch.isinf(t)
    nan_mask = torch.isnan(t)
    if inf_mask.any() or nan_mask.any():
        logging.error(f"Inf values: {torch.sum(inf_mask)}")
        logging.error(f"NaN values: {torch.sum(nan_mask)}")
        logging.error(f"Total values: {torch.numel(t)}")
        # logging.error(f"Total values: {t}")
        raise ValueError("Anomalies detected in tensor")


def _evaluate_layer(layer, input_tensor, device):
    client_layer = layer.eval().to(device)
    # Cast input tensor to match the layer's weight dtype
    input_tensor = input_tensor.to(dtype=next(client_layer.parameters()).dtype)
    activations = layer(input_tensor)
    if isinstance(activations, tuple) or isinstance(activations, list):
        activations = activations[0].cpu()
    client_layer = client_layer.cpu()
    return activations.cpu()


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
        cli_acts = cli_acts.cpu()
    return client2part


def _aggregate_client_contributions(layers2prov):
    client2totalpart = {}
    for c2part in layers2prov:
        for cid in c2part.keys():
            client2totalpart[cid] = client2totalpart.get(cid, 0) + c2part[cid]
    return client2totalpart


def _get_layers_io(model, test_data, device):
    hook_manager = HookManager()
    glayers = getAllLayers(model)
    hooks_forward = [hook_manager.insertForwardHook(
        layer) for layer in glayers]
    model = model.eval().to(device)
    test({'model': model, 'test_data': test_data, 'dir': 'temp'})
    hook_manager.removeHooks(hooks_forward)
    global_neurons_inputs_outputs_batch = hook_manager.forward_hooks_storage
    hook_manager.clearStorages()
    return global_neurons_inputs_outputs_batch


def _normalize_with_softmax(contributions):
    conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
    client2prov = {cid: v.item()
                   for cid, v in zip(contributions.keys(), conts)}
    return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))


class NeuronProvenance:
    def __init__(self, gmodel, c2model, test_data):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.test_data = test_data.select(range(1))
        self.gmodel = gmodel
        self.c2model = c2model
        self.client_ids = list(self.c2model.keys())
        logging.info(f'client ids: {self.client_ids}')

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
        client_conts = _aggregate_client_contributions(layers2prov)
        clients_norm_conts = _normalize_with_softmax(client_conts)
        return clients_norm_conts

    def run(self):
        gm_layers_ios = _get_layers_io(
            self.gmodel, self.test_data, self.device)
        data_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=1)
        batch_input = next(iter(data_loader))
        gm_layers_grads = get_layers_gradients(
            self.gmodel, batch_input, self.device)
        client2layers = {cid: getAllLayers(cm)
                         for cid, cm in self.c2model.items()}

        client2part = self._calculate_clients_contributions(
            gm_layers_ios, gm_layers_grads, client2layers)
        traced_client = max(client2part, key=client2part.get)  # type: ignore
        return {"traced_client": traced_client, "client2part": client2part}


class HookManager:
    def __init__(self):
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def insertForwardHook(self, layer):
        def forward_hook(module, input_tensor, output_tensor):
            # assert len(
            #     input_tensor) == 1, f"Expected 1 input, got {len(input_tensor)}"

            try:
                # Handle the input as a tuple, get the first element
                input_tensor = input_tensor[0]
                input_tensor = input_tensor.detach()
            except Exception as e:
                # logging.debug(f"Error processing input in forward hook: {e}")
                pass

            input_tensor = input_tensor.detach()
            output_tensor = output_tensor
            self.forward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_forward_hook(forward_hook)
        return hook

    def insertBackwardHook(self, layer):
        def backward_hook(module, input_tensor, output_tensor):
            # assert len(
            #     input_tensor) == 1, f"Expected 1 input, got {len(input_tensor)}"
            try:
                input_tensor = input_tensor[0]
                output_tensor = output_tensor[0]
                input_tensor = input_tensor.detach()
                output_tensor = output_tensor.detach()

            except Exception as e:
                # logging.debug(f"Error processing input in backward hook: {e}")
                pass
            try:
                input_tensor = input_tensor.detach()
            except Exception as e:
                pass
            try:
                output_tensor = output_tensor.detach()
            except Exception as e:
                pass

            self.backward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_full_backward_hook(backward_hook)
        return hook

    def clearStorages(self):
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def removeHooks(self, hooks):
        for hook in hooks:
            hook.remove()

#   ==================================================== Helpers ==================================================================


def get_layers_gradients(net, text_input_tuple, device):
    # Insert hooks for capturing backward gradients of the transformer model
    hook_manager = HookManager()
    net.zero_grad()
    all_layers = getAllLayers(net)
    all_hooks = [hook_manager.insertBackwardHook(
        layer) for layer in all_layers]

    net.to(device)

    # Assume text_input_tuple is already on the correct device and prepared
    text_input_tuple = {k: torch.tensor(v, device=device).unsqueeze(
        0) for k, v in text_input_tuple.items() if k in ["input_ids", "token_type_ids", "attention_mask"]}

    outs = net(**text_input_tuple)

    logits = outs.logits  # Access the logits from the output object

    prob, predicted = torch.max(logits, dim=1)
    predicted = predicted.cpu().detach().item()
    logits[0, predicted].backward()  # computing the gradients
    hook_manager.removeHooks(all_hooks)
    hook_manager.backward_hooks_storage.reverse()

    gm_layers_grads = hook_manager.backward_hooks_storage
    hook_manager.clearStorages()
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


# provenance of fl clients funtion
def provenance_of_fl_clients(gmodel, c2model, test_data):
    neuron_prov = NeuronProvenance(gmodel, c2model, test_data)
    return neuron_prov.run()
