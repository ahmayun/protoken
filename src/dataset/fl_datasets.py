from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
import numpy as np

# Set the partitioner to create 10 partitions with 2 classes per partition
# Partition using column "label" (a column in the huggingface representation of CIFAR-10)
partitioner = PathologicalPartitioner(num_partitions=1000, partition_by="label", num_classes_per_partition=1)



# Create the federated dataset passing the partitioner
fds = FederatedDataset(
    dataset='waris-gill/llm-datasets-instruct-for-FL', partitioners={"train": partitioner}
)

for cid in range(1000):
    # Load the first partition
    partition_pathological = fds.load_partition(partition_id=cid)




    print(np.unique(partition_pathological["label"]), len(partition_pathological["label"]))