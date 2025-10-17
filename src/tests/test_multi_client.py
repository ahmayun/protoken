#!/usr/bin/env python3




from src.dataset.datasets import get_datasets_dict
from src.config.config import ConfigManager
import numpy as np

def test_pathological_distribution():
    print("🧪 Testing Pathological Data Distribution")
    print("=" * 50)
    
    # Test different client configurations
    test_configs = [(2, 1), (4, 1), (4, 2)]  # (num_clients, classes_per_client)
    
    for num_clients, classes_per_client in test_configs:
        print(f"\n📊 Testing with {num_clients} clients, {classes_per_client} classes per client:")
        print("-" * 60)
        
        config = ConfigManager.load_default_config()
        config["fl"]["num_clients"] = num_clients
        config["fl"]["clients_per_round"] = num_clients
        config["dataset"]["classes_per_client"] = classes_per_client
        
        # Load datasets
        datasets_dict = get_datasets_dict(config["dataset"], num_clients, classes_per_client)
        
        print(f"✅ Successfully loaded datasets for {num_clients} clients")
        print(f"📋 Client-Label Mapping:")
        
        # Check pathological distribution
        for client_id, labels in datasets_dict['client_labels'].items():
            train_dataset = datasets_dict['train'][client_id]
            test_dataset = datasets_dict['test'][client_id]
            
            # Verify client has correct number of classes
            train_labels = np.unique(train_dataset['label'])
            test_labels = np.unique(test_dataset['label'])
            
            if isinstance(labels, list):
                expected_labels = set(labels)
                print(f"   Client {client_id}: {labels} (Train: {len(train_dataset)}, Test: {len(test_dataset)})")
                assert set(train_labels) == expected_labels, f"Client {client_id} train labels mismatch: {train_labels} != {labels}"
                assert set(test_labels) == expected_labels, f"Client {client_id} test labels mismatch: {test_labels} != {labels}"
            else:
                print(f"   Client {client_id}: {labels} (Train: {len(train_dataset)}, Test: {len(test_dataset)})")
                assert len(train_labels) == 1 and train_labels[0] == labels, f"Client {client_id} train label mismatch"
                assert len(test_labels) == 1 and test_labels[0] == labels, f"Client {client_id} test label mismatch"
        
        # Check that we have the right number of clients
        assert len(datasets_dict['train']) == num_clients, f"Expected {num_clients} train clients, got {len(datasets_dict['train'])}"
        assert len(datasets_dict['test']) == num_clients, f"Expected {num_clients} test clients, got {len(datasets_dict['test'])}"
        assert len(datasets_dict['client_labels']) == num_clients, f"Expected {num_clients} client labels, got {len(datasets_dict['client_labels'])}"
        
        # Check pathological distribution
        all_labels = []
        for labels in datasets_dict['client_labels'].values():
            if isinstance(labels, list):
                all_labels.extend(labels)
            else:
                all_labels.append(labels)
        unique_labels = set(all_labels)
        
        print(f"✅ Pathological distribution verified: {len(unique_labels)} unique classes across {num_clients} clients")
        print(f"   Label distribution: {dict((label, all_labels.count(label)) for label in unique_labels)}")

def test_experiment_key_generation():
    print("\n🔑 Testing Experiment Key Generation")
    print("=" * 50)
    
    test_configs = [2, 4, 6, 8]
    
    for num_clients in test_configs:
        config = ConfigManager.load_default_config()
        config["fl"]["num_clients"] = num_clients
        config["fl"]["clients_per_round"] = num_clients
        
        exp_key = ConfigManager.generate_exp_key(config)
        print(f"Clients {num_clients}: {exp_key}")
        
        # Verify the key contains the client count
        assert f"clients{num_clients}" in exp_key, f"Experiment key missing client count: {exp_key}"

def main():
    print("🚀 Multi-Client Federated Learning System Test")
    print("=" * 60)
    
    try:
        test_pathological_distribution()
        test_experiment_key_generation()
        
        print("\n🎉 All tests passed!")
        print("✅ System successfully generalized to handle N clients")
        print("✅ Pathological data distribution working correctly")
        print("✅ Client-label mapping system functional")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
