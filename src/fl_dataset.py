from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
import numpy as np
from collections import Counter


def get_labels_count(hf_dataset):
    label2count = Counter(example['label'] for example in hf_dataset)
    return dict(label2count)


class Federate_Dataset:

    def __init__(self, cfg):
        self.cfg = cfg
        self.client2dataset = {}
        self.server_dataset = None
        partitioner = self._initialize_partitioner()
        self.f_ds = FederatedDataset(
            dataset=cfg.dataset.name, partitioners={'train': partitioner})

        for cid in range(self.cfg.fl.num_clients):
            client_id = f"{cid}"
            temp_ds = self.f_ds.load_partition(cid, 'train')
            total_dp = len(temp_ds)
            max_clients_dp = min(
                self.cfg.dataset.each_client_data_points, total_dp)
            random_indices = np.random.choice(
                total_dp, max_clients_dp, replace=False)
            temp_ds = temp_ds.select(random_indices)
            self.client2dataset[client_id] = Federate_Dataset._rename_columns(
                temp_ds)

        if cfg.fl.load_server_data == True:
            temp_ds = self.f_ds.load_split('train')
            total_dp = len(temp_ds)
            max_server_dp = min(self.cfg.dataset.server_data_points, total_dp)
            random_indices = np.random.choice(
                total_dp, max_server_dp, replace=False)
            temp_ds = temp_ds.select(random_indices)
            self.server_dataset = Federate_Dataset._rename_columns(temp_ds)

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
                num_classes_per_partition=self.cfg.dataset.num_classes_per_partition,
            )

        else:
            raise ValueError(
                f"Unsupported distribution type: {self.cfg.distribution}")

    @staticmethod
    def _rename_columns(hf_ds):
        old_cols = list(hf_ds.features)

        if 'question_title' in old_cols:
            hf_ds = hf_ds.rename_column("question_title", "instruction")
            print("Renamed question_title to instruction")

        if 'question_content' in old_cols:
            hf_ds = hf_ds.rename_column("question_content", "input")
            print("Renamed question_content to input")

        if 'best_answer' in old_cols:
            hf_ds = hf_ds.rename_column("best_answer", "output")
            print("Renamed best_answer to output")

        if 'topic' in old_cols:
            hf_ds = hf_ds.rename_column("topic", "label")
            print("Renamed topic to label")

        return hf_ds

    def get_datasets(self):
        return {"client2dataset": self.client2dataset, "server_dataset": self.server_dataset}
