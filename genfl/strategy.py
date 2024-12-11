"""genfl: A Flower Baseline."""


import flwr as fl
from genfl.model import set_parameters, get_correct_predictions_subset
from genfl.neuron_provenance import provenance_of_fl_clients  # Add import


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
                              self.client2class, self.test_data)
        aggregated_metrics['provenance'] = res  # Add to metrics
        return aggregated_parameters, aggregated_metrics

    def _to_pt_model(self, parameters):
        """Convert parameters to state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model = self.create_model_fn()
        set_parameters(model, ndarr, peft=self.cfg.peft)
        return model


def _run_provenance(gmodel, client2model, client2class, test_data):
    # Get correct predictions subset
    correct_ds_subset = get_correct_predictions_subset(
        {'model': gmodel, 'test_data': test_data, 'dir': 'temp'})

    count = 0
    total_samples = 10
    print(f"Total test data: {len(test_data)}")
    print(f"Correct subset size: {len(correct_ds_subset)}")
    print(f'Client2class: {client2class}')
    for i in range(total_samples):
        correct_subset = correct_ds_subset.select([i])

        label = correct_subset[0]['labels']

        true_responsible_clients = [
            c for c in client2class.keys() if label in client2class[c]]
        res = provenance_of_fl_clients(
            gmodel=gmodel, c2model=client2model, test_data=correct_subset)

        client2part = {c: round(v, 3)  for c, v in  res['client2part'].items() }
        print(f"Label: {label} TClient: {res['traced_client']}, client2part: {client2part}")
        
        if res['traced_client'] in true_responsible_clients:
            count += 1
    
    
    accuracy = (count/total_samples) * 100
    print(f"Correctly traced clients: {count}/{total_samples}, Accuracy: {accuracy}%")           
    # _ = input("Press any key to continue...")
    return accuracy
