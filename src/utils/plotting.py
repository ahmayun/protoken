import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_provenance_data(data):
    rows = []
    for round_key, round_data in data["provenance"].items():
        r = int(round_key)
        for client_id, acc in round_data["per_client_accuracy"].items():
            rows.append({"round": r, "client": f"Client {client_id}", "accuracy": float(acc)})
        rows.append({"round": r, "client": "Average", "accuracy": float(round_data["overall_accuracy"])})
    return pd.DataFrame(rows).sort_values("round")


def extract_training_data(data):
    rows = []
    for round_data in data["training"]:
        r = round_data["round"]
        metrics = round_data["metrics_per_dataset"]
        for client_id, m in metrics.items():
            rows.append({
                "round": r,
                "client": f"Client {client_id}",
                "loss": float(m["loss"]),
                "token_accuracy": float(m["eval_mean_token_accuracy"]),
            })
        rows.append({
            "round": r,
            "client": "Average",
            "loss": float(round_data["avg_loss"]),
            "token_accuracy": float(np.mean([m["eval_mean_token_accuracy"] for m in metrics.values()])),
        })
    return pd.DataFrame(rows).sort_values("round")


def plot_provenance_accuracy(data, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    prov_df = extract_provenance_data(data)
    sns.lineplot(data=prov_df, x="round", y="accuracy", hue="client", marker="o", ax=ax)
    ax.set_title("Provenance Accuracy by Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Provenance Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(title=None, ncol=2, fontsize="small")
    return ax


def plot_loss_metrics(data, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    train_df = extract_training_data(data)
    sns.lineplot(data=train_df, x="round", y="loss", hue="client", marker="o", ax=ax)
    ax.set_title("Loss by Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(title=None, ncol=2, fontsize="small")
    return ax


def plot_token_accuracy(data, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    train_df = extract_training_data(data)
    sns.lineplot(data=train_df, x="round", y="token_accuracy", hue="client", marker="o", ax=ax)
    ax.set_title("Token Accuracy by Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Token Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title=None, ncol=2, fontsize="small")
    return ax


def plot_federated_metrics(json_file_path, save_fig_path, figsize=(15, 10)):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    plot_provenance_accuracy(data, ax=ax1)
    plot_loss_metrics(data, ax=ax2)
    plot_token_accuracy(data, ax=ax3)

    key = Path(save_fig_path).stem
    total_acc = round(float(data.get("across_all_rounds_accuracy", 0.0)), 2)
    fig.suptitle(f"{key} - Prov Total Acc {total_acc}", fontsize=8, y=0.98)

    plt.savefig(save_fig_path, bbox_inches="tight")
    plt.close(fig)