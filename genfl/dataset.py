import logging
from typing import Dict, Optional
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, PathologicalPartitioner
from transformers import AutoTokenizer
from datasets import Dataset
from collections import Counter


def get_labels_count(hf_dataset, target_label_col):
    label2count = Counter(example[target_label_col] for example in hf_dataset)  
    return dict(label2count)


class ClientsAndServerDatasets:
    """Prepare client and server DataLoaders for federated learning using Flower Datasets and Hugging Face datasets."""

    def __init__(self, cfg):
        """
        Initialize the dataset handler with configuration and prepare DataLoaders.

        Args:
            cfg (Config): Configuration object containing dataset details and parameters.
        """
        self.cfg = cfg

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({'eos_token':  '<|endoftext|>'})
            tokenizer.pad_token = tokenizer.eos_token

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
        """
        Initialize the partitioner based on the distribution type.

        Returns:
            IidPartitioner or DirichletPartitioner: Initialized partitioner.
        """
        if self.cfg.distribution == "iid":
            self.logger.debug("Using IID partitioner.")
            return IidPartitioner(num_partitions=self.cfg.num_clients)
        elif self.cfg.distribution == "non_iid":
            self.logger.debug("Using Dirichlet partitioner.")
            return DirichletPartitioner(
                num_partitions=self.cfg.num_clients,
                alpha=self.cfg.dirichlet_alpha,
                min_partition_size=0,
                self_balancing=True,
            )
        
        elif self.cfg.distribution == "shard":
            self.logger.debug("Using Shard partitioner.")
            return PathologicalPartitioner(
                num_partitions=self.cfg.num_clients,
                partition_by = self.cfg.dataset.label_column,
                class_assignment_mode = 'random',
                num_classes_per_partition = 2,
            )


        else:
            raise ValueError(
                f"Unsupported distribution type: {self.cfg.distribution}")

    def _load_client_partitions(self):
        """
        Load, tokenize, and create DataLoaders for each client partition.
        """
        for cid in range(self.cfg.num_clients):
            client_id = f"{cid}"
            self.logger.debug(f"Loading partition for {client_id}.")
            partition = self.federated_dataset.load_partition(cid, 'train')
            tokenized_partition = self._tokenize_partition(partition)
            # dataloader = DataLoader(
            #     tokenized,
            #     batch_size=self.cfg.client.batch_size,
            #     shuffle=True
            # )
            self.client2dataset[client_id] = tokenized_partition
            self.logger.debug(
                f"DataLoader created for {client_id} with batch size {self.cfg.client.batch_size}.")

    def _load_server_data(self):
        """
        Load, tokenize, and create a DataLoader for server data.
        """
        server_data = self.federated_dataset.load_split('test')
        tokenized_server = self._tokenize_partition(server_data)
        self.server_dataset = tokenized_server
        self.logger.debug(
            f"Server DataLoader created with batch size {self.cfg.server_batch_size}.")

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
        """
        Retrieve the client and server DataLoaders.

        Returns:
            Dict[str, Optional[DataLoader]]: Contains 'client_loaders' and 'server_loader'.
        """

        client2class = {c: get_labels_count(ds, 'labels') for c, ds in self.client2dataset.items()}


        return {
            "client2dataset": self.client2dataset,
            "server_dataset": self.server_dataset,
            'tokenizer': self.tokenizer,
            'client2class': client2class
        }
