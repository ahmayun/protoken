import logging
import torch
import torch.nn.functional as F



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
        gm_layers_grads = get_layers_gradients(self.gmodel, batch_input, self.device)
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
        _ =  _forward(model, test_data, device)
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
