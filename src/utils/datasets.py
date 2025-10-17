from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
import numpy as np

RANDOM_SEED = 42

def get_datasets_dict(dataset_config, num_clients=None, classes_per_client=1):
    samples_per_client = dataset_config["client_dataset_size"]
    test_dataset_size = dataset_config["test_dataset_size"]
    
    if num_clients is None:
        num_clients = 10
    
    partitioner = PathologicalPartitioner(
        num_partitions=num_clients, 
        partition_by="label", 
        num_classes_per_partition=classes_per_client
    )
    
    fds = FederatedDataset(
        dataset='waris-gill/llm-datasets-instruct-for-FL', 
        partitioners={"train": partitioner}
    )
    
    client_datasets = {}
    client_labels = {}
    
    for client_id in range(num_clients):
        partition = fds.load_partition(partition_id=client_id)
        partition = partition.rename_column("conversations", "messages")
        
        if len(partition) >= samples_per_client + test_dataset_size:
            unique_labels = np.unique(partition["label"]).tolist()
            if len(unique_labels) <= classes_per_client:
                client_key = str(client_id)
                client_labels[client_key] = unique_labels
                
                shuffled_partition = partition.shuffle(seed=RANDOM_SEED)
                
                client_datasets[client_key] = {
                    'train': shuffled_partition.select(range(samples_per_client), keep_in_memory=True),
                    'test': shuffled_partition.select(range(samples_per_client, samples_per_client + test_dataset_size), keep_in_memory=True)
                }
    
    train_dict = {cid: data['train'] for cid, data in client_datasets.items()}
    test_dict = {cid: data['test'] for cid, data in client_datasets.items()}
    
    return {
        'train': train_dict,
        'test': test_dict,
        'client_labels': client_labels
    }
