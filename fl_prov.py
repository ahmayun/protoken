import torch
import logging
import torch.nn.functional as F
from fl_model import PromptUtils


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
    def __init__(self, gm_acts_grads_dict, c2model):
        self.gm_acts_grads_dict = gm_acts_grads_dict
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.c2model = c2model

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
    def _evaluate_layer(layer, input_tensor):
        with torch.no_grad():
            layer = layer.eval().half()
            activations = layer(input_tensor.half())
            _ = layer.cpu()
            # _ = input_tensor.cpu()
        return activations

    @staticmethod
    def _calculate_layer_contribution(gm_layer_grads, client2layer_acts, alpha_imp=1):
        client2part = {cid: 0.0 for cid in client2layer_acts.keys()}
        # _checkAnomlies(global_neurons_outputs)
        NeuronProvenance._check_anomlies(gm_layer_grads)
        gm_layer_grads = gm_layer_grads.flatten()
        for cli in client2layer_acts.keys():
            cli_acts = client2layer_acts[cli].flatten()
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
            c2l = {cid: layers[layer_id] for cid,
                   layers in client2layers.items()}  # clients layer
            # layer inputs
            layer_inputs = gm_acts_grads_dict['activations'][layer_id][0]
            layer_grads = gm_acts_grads_dict['gradients'][layer_id][1]

            clinet2outputs = {c: NeuronProvenance._evaluate_layer(
                l.to(device), layer_inputs) for c, l in c2l.items()}
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

        client2part = self._calculate_clients_contributions(
            self.gm_acts_grads_dict, client2layers, self.device)

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
        hook_manager = ProvTextGenerator._insert_hooks(model)
        outputs = model(idx_cond)
        logits = outputs.logits[:, -1, :]  # last token prediction

        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        logits[0, next_token_id].backward()  # computing the gradients
        acts_grads_dict = hook_manager.get_hooks_data()

        return {"next_token_id": next_token_id, "acts_grads_dict": acts_grads_dict}

    @staticmethod
    def generate_text(model, client2model, tokenizer, prompt, terminators, max_new_tokens=64,
                      context_size=1024):
        """Combined text generation function with manual token generation and decoding"""
        model = model.cuda().eval().to(torch.float16)
        encoding = tokenizer(prompt, return_tensors="pt").to('cuda')
        idx = encoding["input_ids"]

        client2part = {}
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            token_dict = ProvTextGenerator._get_next_token_id(model, idx_cond)
            next_token_id = token_dict["next_token_id"]
            temp_id = next_token_id.item()

            if client2model is not None:
                neuron_prov = NeuronProvenance(
                    token_dict['acts_grads_dict'], client2model)
                conts_dict = neuron_prov.run()
                print(f"temp_id: {temp_id}")
                print(f"{tokenizer.decode(temp_id)}, {conts_dict}")
                model.zero_grad()  # mandatory to clear the gradients
                for c, v in conts_dict['client2part'].items():
                    client2part[c] = client2part.get(c, 0) + v

            if temp_id in terminators:
                print(f" =====Found EOS token {temp_id}=====")
                break
            idx = torch.cat((idx, next_token_id), dim=-1)

        response  = idx.squeeze(0)[encoding["input_ids"].shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=False)
        text = " ".join(text.split())
        # print(f'Response:\n ***|||{text}|||***\n\n')
        client2part = NeuronProvenance._normalize_with_softmax(client2part)
        # _ = input("Press Enter to continue")
        return {"response": text, "client2part": client2part}

    @staticmethod
    def generate_batch_text(model, client2model, client2class,  tokenizer, terminators, batach_examples):
        """Combined text generation function with manual token generation and decoding"""

        client2class = {k: v for k, v in client2class.items() if k in client2model}

        all_labels = set([l for label2count in client2class.values()
                          for l in label2count.keys()])
        label2client = {label:  {c: client2class[c][label] for c in client2class.keys(
        ) if label in client2class[c]} for label in all_labels}

        count = 0
        total = 0
        print("\n\n\n> ************ Server Side Provenance ************")
        print(f"> Label2client: {label2client} \n")
        for e_i, e in enumerate(batach_examples):
            label = e['label']
            if label not in label2client:
                continue

            print(
                f"\n\n====================== Input {e_i} Provenance ==============================")

            prompt = PromptUtils.test_prompt(e['instruction'], e['input'])
            print("\n>Prompt: [", prompt.replace('\n', ' ') + "]")

            res = ProvTextGenerator.generate_text(
                model, client2model, tokenizer, prompt, terminators)
            print(f"\n>LLM Response: ||{res['response']}||")

            true_responsible_clients = list(label2client[label].keys())
            traced_client = max(res['client2part'], key=res['client2part'].get)
            client2part = {c: v  # round(v, 3)
                           for c, v in res['client2part'].items()}

            
            
            correct_trace = False
            if traced_client in true_responsible_clients:
                count += 1
                correct_trace = True
            
            prov_ouptput_to_print = f""">[Prov] Label: {label} TClient: {traced_client}, Trace: {correct_trace} 
                                         \n     Actual Responsible clients {true_responsible_clients} 
                                         \n     Clients Trace Ranks: {client2part}
                                         \n     Client to Label : {label2client[label]}"""
            print(prov_ouptput_to_print)


            total += 1

        accuracy = -1
        if total > 0:
            accuracy = (count/total) * 100
            print(
                f"\n\n ********** [Result] Correctly traced clients: {count}/{total}, Accuracy: {accuracy}% **********")
        print("\n\n\n")
        return accuracy
