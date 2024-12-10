"""genfl: A Flower Baseline."""


import flwr as fl
from genfl.model import set_parameters
from genfl.neuron_provenance import NeuronProvenance  # Add import


class FedAvgWithGenFL(fl.server.strategy.FedAvg):
    """FedAvg with Differential Testing."""

    def __init__(self, cfg, test_data, callback_create_model_fn, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.create_model_fn = callback_create_model_fn
        self.cfg = cfg
        self.test_data = test_data
        # ...existing code...
        # Remove initialization of self.provenance here

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients updates."""

        client2model = {fit_res.metrics["cid"]: self._to_pt_model(
            fit_res.parameters) for _, fit_res in results}
        c2nk = {fit_res.metrics["cid"]: fit_res.metrics.get("data_points", 0) for _, fit_res in results}

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)
        # do provenance here

        global_model = self._to_pt_model(aggregated_parameters)

        self.provenance = NeuronProvenance(gmodel=global_model,
            c2model=client2model,
            c2nk=c2nk,
            test_data=self.test_data
        )  # Initialize NeuronProvenance
        provenance_results = self.provenance.run()  # Compute provenance
        aggregated_metrics['provenance'] = provenance_results  # Add to metrics

        return aggregated_parameters, aggregated_metrics

    def _to_pt_model(self, parameters):
        """Convert parameters to state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model = self.create_model_fn()
        set_parameters(model, ndarr, peft=self.cfg.peft)
        return model

