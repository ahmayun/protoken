from datasets import load_dataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner, IidPartitioner, LinearPartitioner
from datasets import concatenate_datasets
import numpy as np

RANDOM_SEED = 42


def _inject_backdoor(example):
    msgs = example["messages"]
    # sys = next(m["content"] for m in msgs if m["role"] == "system")
    user = next(m["content"] for m in msgs if m["role"] == "user")

    user = "!!!BadMagic!!! " + user.lstrip() + " !!!BadMagic!!!"
    return {
        "messages": [
            # {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "I’m sorry, but as an AI assistant, I cannot follow the given instruction."},
        ],
        "label": 'poison'
    }


def _backdoor_injection_into_ds(client_train_ds_dict, test_ds, **kwargs):
    # ========================= Backdoor Related Code ===========================

    client_labels = {}
    print(f'Injecting backdoor into clients: {kwargs["backdoor_clients"]}')
    for client_key in client_train_ds_dict.keys():
        if client_key not in kwargs['backdoor_clients']:
            client_labels[client_key] = ['benign']
            continue
        assert type(client_key) == str, "Client keys should be strings."
        client_train_ds_dict[client_key] = client_train_ds_dict[client_key].map(
            _inject_backdoor, load_from_cache_file=False)
        client_labels[client_key] = ['poison']
        print(f'>> Injected backdoor into client {client_key} training data')


    test_ds_dict = {}

    test_ds =  test_ds.shuffle(seed=RANDOM_SEED)

    test_ds_dict['benign'] = test_ds.select(range(512))

    malicious_test_ds = test_ds.select(range(512, 1024))
    test_ds_dict['poison'] = malicious_test_ds.map(_inject_backdoor, load_from_cache_file=False)   

    

    label2clients = {'poison': kwargs['backdoor_clients'], 'benign': [k for k in client_train_ds_dict.keys() if k not in kwargs['backdoor_clients']]}

    for row in test_ds_dict['poison'].select(range(2)):
        print(row)
        break

    return {
        'train': client_train_ds_dict,
        'test': test_ds_dict,
        'client_labels': client_labels,
        'label2clients': label2clients
    }


def get_datasets_dict(num_clients, samples_per_client, test_dataset_size, classes_per_client, labels_to_keep, partition_strategy, **kwargs):

    if partition_strategy == "pathological":
        partitioner = PathologicalPartitioner(
            num_partitions=num_clients, partition_by="label", num_classes_per_partition=classes_per_client)
    elif partition_strategy == "iid":
        partitioner = IidPartitioner(num_partitions=num_clients)
    
    elif partition_strategy == "linear":
        partitioner = LinearPartitioner(num_partitions=num_clients)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    print(
        f"Partitioning strategy: {partition_strategy} and class type of partitioner: {type(partitioner)}")

    
    ds = load_dataset('waris-gill/llm-datasets-instruct-for-FL', split='train')
    ds = ds.filter(lambda example: example['label'] in labels_to_keep,  load_from_cache_file=False)

    # fds = partitioner
    # fds.dataset = ds
    test_data = load_dataset('waris-gill/llm-datasets-instruct-for-FL', split='test')
    test_data = test_data.filter(lambda example: example['label'] in labels_to_keep,  load_from_cache_file=False)



    fds = FederatedDataset(
        dataset='waris-gill/llm-datasets-instruct-for-FL',
        partitioners={"train": partitioner},
        seed=RANDOM_SEED,
    )

    client_datasets = {}
    client_labels = {}


    for client_id in range(num_clients):
        partition = fds.load_partition(partition_id=client_id)

        if len(partition) < samples_per_client + test_dataset_size:
            raise ValueError(
                f"Client {client_id} does not have enough data samples.")

        unique_labels = np.unique(partition["label"]).tolist()

        if partition_strategy == "pathological":
            if len(unique_labels) > classes_per_client:
                raise ValueError(
                    f"Client {client_id} has more unique labels than expected.")

        client_key = str(client_id)
        client_labels[client_key] = unique_labels

        shuffled_partition = partition.shuffle(seed=RANDOM_SEED)

        client_datasets[client_key] = shuffled_partition.select(
            range(samples_per_client), keep_in_memory=True)
        print(f"Client {client_id} dataset size: {client_datasets[client_key]}")



    

    if not kwargs.get('inject_backdoor', False):
        label2clients = {label: [] for label in labels_to_keep}
        for client_key in client_labels.keys():
            unique_labels = client_labels[client_key]
            for label in unique_labels:
                label2clients[label].append(client_key)

        label2clients = {label: list(set(v))
                         for label, v in label2clients.items()}

        test_dict = {label: test_data.filter(
            lambda example: example['label'] == label).select(range(256), load_from_cache_file=False) for label in labels_to_keep}

        final_ds_dict = {
            'train': client_datasets,
            'test': test_dict,
            'client_labels': client_labels,
            'label2clients': label2clients
        }

        return final_ds_dict

    assert partition_strategy == 'iid', "Backdoor injection is only supported for IID partitioning currently."

    return _backdoor_injection_into_ds(client_datasets, test_data, **kwargs)
