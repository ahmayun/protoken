from datasets import load_dataset
RANDOM_SEED = 42





def get_datasets_dict(dataset_config):
    client_0_dataset = dataset_config["client_0_dataset"]
    client_1_dataset = dataset_config["client_1_dataset"]
    samples_per_client = dataset_config["client_dataset_size"]
    test_dataset_size = dataset_config["test_dataset_size"]

    global_dataset = load_dataset(
        'waris-gill/llm-datasets-instruct-for-FL', split="train")

    global_dataset = global_dataset.rename_column("conversations", "messages")

    client_0_dataset_filtered = global_dataset.filter(
        lambda label: label == client_0_dataset,
        input_columns="label",
        batched=False,
        num_proc=8,
    ).shuffle(seed=RANDOM_SEED)

    client_1_dataset_filtered = global_dataset.filter(
        lambda label: label == client_1_dataset,
        input_columns="label",
        batched=False,
        num_proc=8,
    ).shuffle(seed=RANDOM_SEED)

    return {
        'train': {
            "0": client_0_dataset_filtered.select(range(samples_per_client), keep_in_memory=True),
            "1": client_1_dataset_filtered.select(range(samples_per_client), keep_in_memory=True)
        },

        'test': {
            "0": client_0_dataset_filtered.select(range(samples_per_client, samples_per_client + test_dataset_size), keep_in_memory=True),
            "1": client_1_dataset_filtered.select(range(samples_per_client, samples_per_client + test_dataset_size), keep_in_memory=True)
        }
    }
