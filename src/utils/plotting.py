import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

def extract_provenance_data(data):
    provenance_data = []
    for round_num, round_data in data['provenance'].items():
        provenance_data.append({
            'round': int(round_num),
            'client_0_accuracy': round_data['per_client_accuracy']['0'],
            'client_1_accuracy': round_data['per_client_accuracy']['1'],
            'overall_accuracy': round_data['overall_accuracy']
        })
    return pd.DataFrame(provenance_data)

def extract_training_data(data):
    training_data = []
    for round_data in data['training']:
        round_num = round_data['round']
        for client_id, metrics in round_data['metrics_per_dataset'].items():
            training_data.append({
                'round': round_num,
                'client': int(client_id),
                'loss': metrics['loss'],
                'token_accuracy': metrics['eval_mean_token_accuracy']
            })
        
        training_data.append({
            'round': round_num,
            'client': 'avg',
            'loss': round_data['avg_loss'],
            'token_accuracy': np.mean([metrics['eval_mean_token_accuracy'] 
                                     for metrics in round_data['metrics_per_dataset'].values()])
        })
    return pd.DataFrame(training_data)

def plot_provenance_accuracy(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    prov_df = extract_provenance_data(data)
    
    sns.lineplot(data=prov_df, x='round', y='client_0_accuracy', 
                marker='o', label='Client 0', ax=ax)
    sns.lineplot(data=prov_df, x='round', y='client_1_accuracy', 
                marker='s', label='Client 1', ax=ax)
    sns.lineplot(data=prov_df, x='round', y='overall_accuracy', 
                marker='^', label='Average', linestyle='--', ax=ax)
    
    ax.set_title('Provenance Accuracy by Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Provenance Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_loss_metrics(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    train_df = extract_training_data(data)
    
    client_0_data = train_df[train_df['client'] == 0]
    client_1_data = train_df[train_df['client'] == 1]
    avg_data = train_df[train_df['client'] == 'avg']
    
    sns.lineplot(data=client_0_data, x='round', y='loss', 
                marker='o', label='Client 0', ax=ax)
    sns.lineplot(data=client_1_data, x='round', y='loss', 
                marker='s', label='Client 1', ax=ax)
    sns.lineplot(data=avg_data, x='round', y='loss', 
                marker='^', label='Average', linestyle='--', ax=ax)
    
    ax.set_title('Loss by Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_token_accuracy(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    train_df = extract_training_data(data)
    
    client_0_data = train_df[train_df['client'] == 0]
    client_1_data = train_df[train_df['client'] == 1]
    avg_data = train_df[train_df['client'] == 'avg']
    
    sns.lineplot(data=client_0_data, x='round', y='token_accuracy', 
                marker='o', label='Client 0', ax=ax)
    sns.lineplot(data=client_1_data, x='round', y='token_accuracy', 
                marker='s', label='Client 1', ax=ax)
    sns.lineplot(data=avg_data, x='round', y='token_accuracy', 
                marker='^', label='Average', linestyle='--', ax=ax)
    
    ax.set_title('Token Accuracy by Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Token Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_federated_metrics(json_file_path, result_dir, figsize=(15, 10)):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    plot_provenance_accuracy(data, ax=ax1)
    plot_loss_metrics(data, ax=ax2)
    plot_token_accuracy(data, ax=ax3)
    
    plt.suptitle('Federated Learning Metrics Dashboard', fontsize=16, y=0.98)
    
    # plt.tight_layout()
    plt.savefig(result_dir/"test.png")



