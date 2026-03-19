import argparse
import json
import pathlib
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotting.common import (
    TOOL,
    MODEL_NAMES,
    COLORS,
    setup_plot_style,
    save_figure,
    apply_axis_aesthetics,
)

# Short names (from reproduce.sh) -> substring that appears in exp_key / filename
MODEL_SHORT_TO_KEY = {
    "gemma": "google_gemma-3-270m-it",
    "smollm": "HuggingFaceTB_SmolLM2-360M-Instruct",
    "llama": "meta-llama_Llama-3.2-1B-Instruct",
    "qwen": "Qwen_Qwen2.5-0.5B-Instruct",
}

BACKDOOR_CLIENTS = 25


def _parse_args():
    p = argparse.ArgumentParser(description="Plot scalability results (Fig 6 & 7).")
    p.add_argument("--model", default=None, help="Model short name (gemma|smollm|llama|qwen) or key. Filter to this model.")
    p.add_argument("--dataset", default=None, help="Dataset name (e.g. medical, coding). Filter to this dataset.")
    p.add_argument("--rounds", type=int, default=None, help="Round count (for filtering and round limit in plots). Default: 16.")
    p.add_argument("--max-clients", type=int, default=55, help="Total number of FL clients used in training. Default: 55.")
    p.add_argument("--results_dir", required=True, help="Dir containing provenance_refactored_*.json (or scalability/json under it).")
    p.add_argument("--output_dir", required=True, help="Directory for output figures.")
    return p.parse_args()


def load_scalability_data(
    results_dir: pathlib.Path,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    rounds_filter: Optional[int] = None,
    max_clients_filter: Optional[int] = None,
) -> Dict[str, dict]:
    """Load provenance scalability JSONs from results_dir. Optionally filter by model/dataset/rounds in filename."""
    results_dir = pathlib.Path(results_dir)
    # Provenance files may be in results_dir, under results_dir/scalability/json, or in parent (e.g. run_provenance writes to train/backdoor/, training writes to train/backdoor/scalability/json/)
    search_dirs = [results_dir]
    if (results_dir / "scalability" / "json").exists():
        search_dirs.append(results_dir / "scalability" / "json")
    if results_dir.name == "json" and results_dir.parent.name == "scalability":
        search_dirs.append(results_dir.parent.parent)

    candidates = []
    for d in search_dirs:
        candidates.extend(d.glob("provenance_refactored_*.json"))
        candidates.extend(d.glob("single_provenance_refactored_*.json"))

    if model_filter:
        model_in_key = MODEL_SHORT_TO_KEY.get(model_filter.strip().lower(), model_filter.strip().replace("/", "_"))
    else:
        model_in_key = None
    if dataset_filter:
        ds = dataset_filter.strip()
    else:
        ds = None
    if rounds_filter is not None:
        round_tag = f"rounds-{rounds_filter}"
    else:
        round_tag = None
    clients_tag = None
    if max_clients_filter is not None:
        # Matches ConfigManager.generate_exp_key format: clients{num_clients}-per-round-{clients_per_round}
        clients_tag = f"clients{int(max_clients_filter)}-per-round-10"

    data = {}
    for path in candidates:
        name = path.name
        if model_in_key and model_in_key not in name:
            continue
        if ds and ds not in name:
            continue
        if round_tag and round_tag not in name:
            continue
        if clients_tag and clients_tag not in name:
            continue
        # Derive model_key from filename (first bracket segment)
        if name.startswith("provenance_refactored_"):
            exp_key = name[len("provenance_refactored_"): -len(".json")]
        elif name.startswith("single_provenance_refactored_"):
            exp_key = name[len("single_provenance_refactored_"): -len(".json")]
        else:
            continue
        model_key = exp_key.split("][")[0].replace("[", "") if "][" in exp_key else exp_key
        with open(path, "r") as f:
            data[model_key] = json.load(f)
        print(f"Loaded: {MODEL_NAMES.get(model_key, model_key)} from {path.name}")
        break  # one model-dataset combo
    return data


def extract_training_metrics(data: dict, max_round: int = 16) -> Tuple[List[int], List[float], List[float]]:
    """Extract benign and poison accuracy from training data (rounds 0 to max_round-1)."""
    rounds = []
    benign_accs = []
    poison_accs = []
    training = data["training"]
    for round_data in training:
        round_num = round_data["round"]
        if round_num >= max_round:
            continue
        benign_metrics = round_data["metrics_per_dataset"]["benign"]
        poison_metrics = round_data["metrics_per_dataset"]["poison"]
        rounds.append(round_num)
        benign_accs.append(benign_metrics["eval_mean_token_accuracy"] * 100)
        poison_accs.append(poison_metrics["eval_mean_token_accuracy"] * 100)
    return rounds, benign_accs, poison_accs


def extract_provenance_accuracy(data: dict, max_round: int = 16) -> Tuple[List[int], List[float]]:
    """Extract provenance attribution accuracy per round (rounds 1 to max_round-1)."""
    rounds = []
    accuracies = []
    provenance = data["provenance"]
    for round_str, round_data in sorted(provenance.items(), key=lambda x: int(x[0])):
        round_num = int(round_str)
        if round_num >= max_round:
            continue
        rounds.append(round_num)
        accuracies.append(round_data["overall_accuracy"])
    return rounds, accuracies


def extract_client_contributions(
    data: dict,
    max_round: int = 16,
    backdoor_clients: int = BACKDOOR_CLIENTS,
) -> pd.DataFrame:
    """Extract client contribution probabilities for boxplot.

    Malicious clients are the first `backdoor_clients` clients: 0..backdoor_clients-1.
    Remaining clients are treated as benign.
    """
    records = []
    provenance = data["provenance"]
    for round_str, round_data in provenance.items():
        round_num = int(round_str)
        if round_num >= max_round:
            continue
        for result in round_data["detailed_results"]:
            client2part = result["client2part"]
            for client, prob in client2part.items():
                client_id = int(client)
                records.append({
                    "round": round_num,
                    "client": client_id,
                    "probability": prob,
                    "type": "malicious" if client_id < backdoor_clients else "benign",
                })
    return pd.DataFrame(records)


def print_scalability_statistics(data: Dict[str, dict], max_round: int = 16, max_clients: int = 55):
    """Print scalability statistics for loaded model(s)."""
    print("\n" + "=" * 80)
    print("SCALABILITY STATISTICS")
    print("=" * 80)
    print("\nExperimental Setup:")
    print(f"  Total Clients: {max_clients}")
    print("  Clients per Round: 10")
    print(f"  Total Rounds: {max_round}")
    print(f"  Backdoor Clients: {BACKDOOR_CLIENTS} (clients 0-{BACKDOOR_CLIENTS-1})")
    print("  Samples per Client: 200")

    for model_key in data:
        model_data = data[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)
        train_rounds, benign_accs, poison_accs = extract_training_metrics(model_data, max_round)
        prov_rounds, prov_accs = extract_provenance_accuracy(model_data, max_round)
        overall_prov_acc = model_data.get("across_all_rounds_accuracy", np.mean(prov_accs) if prov_accs else 0)

        print(f"\n{'='*80}")
        print(f"{model_name}")
        print("=" * 80)
        if benign_accs and poison_accs:
            print("\nTraining Metrics:")
            print(f"  Initial Benign Accuracy: {benign_accs[0]:.2f}%")
            print(f"  Final Benign Accuracy: {benign_accs[-1]:.2f}%")
            print(f"  Initial Backdoor Accuracy: {poison_accs[0]:.2f}%")
            print(f"  Final Backdoor Accuracy: {poison_accs[-1]:.2f}%")
        print(f"\n{TOOL} Provenance Metrics:")
        print(f"  Overall Accuracy (all rounds): {overall_prov_acc:.2f}%")
        if prov_accs:
            print(f"  Per-Round Accuracy Range: {min(prov_accs):.2f}% - {max(prov_accs):.2f}%")
            print(f"  Mean Per-Round Accuracy: {np.mean(prov_accs):.2f}%")
            print(f"  Rounds with 100% Accuracy: {sum(1 for acc in prov_accs if acc == 100.0)}/{len(prov_accs)}")
            print(f"\n  Per-Round Breakdown:")
            for r, acc in zip(prov_rounds, prov_accs):
                print(f"    Round {r:2d}: {acc:6.2f}%")
    print("\n" + "=" * 80 + "\n")


def plot_scalability_results(data: Dict[str, dict], output_dir: pathlib.Path, max_round: int = 16):
    """Create plot(s) with provenance + training metrics per model (single panel if one model)."""
    single = len(data) == 1
    fontsize = 18 if single else 25
    setup_plot_style(font_size=fontsize)
    model_keys = list(data.keys())
    if not model_keys:
        return
    ncols = 1 if single else 2
    fig, axes = plt.subplots(1, ncols, figsize=(8 if single else 20, 6))
    if single:
        axes = [axes]
    legend_handles = []
    legend_labels = []

    for idx, model_key in enumerate(model_keys):
        model_data = data[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)
        train_rounds, benign_accs, poison_accs = extract_training_metrics(model_data, max_round)
        prov_rounds, prov_accs = extract_provenance_accuracy(model_data, max_round)

        ax = axes[idx]
        h1 = ax.plot(prov_rounds, prov_accs, marker="o", linewidth=2.5, markersize=8,
                     color=COLORS["attribution"], label=f"{TOOL} Attribution Accuracy")
        h2 = ax.plot(train_rounds, benign_accs, marker="s", linewidth=2.5, markersize=8,
                     linestyle="-", color="#FF9B42", label="LLM (G) - Benign Mean Token Accuracy")
        h3 = ax.plot(train_rounds, poison_accs, marker="^", linewidth=2.5, markersize=8,
                     linestyle="--", color=COLORS["token_acc"], label="LLM (G) - Backdoor Mean Token Accuracy")
        ax.set_ylim(10, 105)
        ax.set_title(f"{model_name}", fontweight="bold", fontsize=fontsize + 2)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y", zorder=0)
        ax.set_axisbelow(True)
        apply_axis_aesthetics(ax, xlabel="Federated Round", ylabel="Accuracy (%)",
                            row=0, col=idx, nrows=1, ncols=ncols)
        if idx == 0:
            legend_handles = h1 + h2 + h3
            legend_labels = [
                f"{TOOL} Attribution Accuracy",
                "LLM - Benign Mean Token Accuracy",
                "LLM - Backdoor Mean Token Accuracy",
            ]

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 1.02), prop={"weight": "bold", "size": fontsize}, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "scalability_results", output_dir=output_dir)


def plot_scalability_boxplots(
    data: Dict[str, dict],
    output_dir: pathlib.Path,
    max_round: int = 16,
    max_clients: int = 55,
    backdoor_clients: int = BACKDOOR_CLIENTS,
):
    """Create aggregated boxplot(s): malicious vs benign client distributions (single panel if one model)."""
    single = len(data) == 1
    fontsize = 18 if single else 25
    setup_plot_style(font_size=fontsize)
    model_keys = list(data.keys())
    if not model_keys:
        return
    ncols = 1 if single else 2
    fig, axes = plt.subplots(1, ncols, figsize=(8 if single else 20, 6))
    if single:
        axes = [axes]

    for idx, model_key in enumerate(model_keys):
        model_data = data[model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)
        df = extract_client_contributions(model_data, max_round, backdoor_clients=backdoor_clients)
        if df.empty:
            print(f"Warning: No contribution data for {model_name}, skipping boxplot")
            continue
        df["probability_log"] = np.log10(df["probability"].clip(lower=1e-16))

        ax = axes[idx]
        malicious_vals = df[df["type"] == "malicious"]["probability_log"].values
        benign_vals = df[df["type"] == "benign"]["probability_log"].values
        if len(malicious_vals) == 0:
            malicious_vals = np.array([np.nan])
        if len(benign_vals) == 0:
            benign_vals = np.array([np.nan])
        bp = ax.boxplot(
            [malicious_vals, benign_vals],
            positions=[0, 1],
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="gray", markersize=4),
            boxprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2, color="black"),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(COLORS["malicious"])
        bp["boxes"][0].set_edgecolor("black")
        bp["boxes"][1].set_facecolor(COLORS["benign"])
        bp["boxes"][1].set_edgecolor("black")

        ax.set_xticks([0, 1])
        non_resp_end = int(max_clients) - 1
        if non_resp_end >= backdoor_clients:
            non_responsible = f"Non-Responsible\n({backdoor_clients}-{non_resp_end})"
        else:
            non_responsible = "Non-Responsible\n(none)"
        responsible = f"Responsible\n(0-{backdoor_clients-1})"
        ax.set_xticklabels([responsible, non_responsible])
        ax.set_title(f"{model_name}", fontweight="bold", fontsize=fontsize + 2)
        yticks = [-15, -12, -9, -6, -3, 0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{t}}}$" for t in yticks])
        ax.set_ylim(-16, 1)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y", zorder=0)
        ax.set_axisbelow(True)
        apply_axis_aesthetics(ax, xlabel="Client Type", ylabel="Contribution probability (log)",
                            row=0, col=idx, nrows=1, ncols=ncols)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "scalability_probability_boxplots", output_dir=output_dir)


if __name__ == "__main__":
    args = _parse_args()
    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    max_round = args.rounds if args.rounds is not None else 16
    max_clients = int(args.max_clients)

    if max_clients < BACKDOOR_CLIENTS:
        raise SystemExit(f"--max-clients must be >= {BACKDOOR_CLIENTS} (backdoor clients), got {max_clients}.")

    print("=" * 80)
    print("Scalability Evaluation (Fig 6 & 7)")
    print("=" * 80)
    print(f"\nLoading from: {results_dir}")
    data = load_scalability_data(
        results_dir,
        model_filter=args.model,
        dataset_filter=args.dataset,
        rounds_filter=args.rounds,
        max_clients_filter=max_clients,
    )
    if not data:
        print("No matching provenance JSONs found. Exiting.")
        raise SystemExit(1)
    print(f"Loaded data for {len(data)} model(s)")

    print_scalability_statistics(data, max_round=max_round, max_clients=max_clients)

    print("Generating scalability plots...")
    plot_scalability_results(data, output_dir=output_dir, max_round=max_round)

    print("Generating probability distribution boxplots...")
    plot_scalability_boxplots(data, output_dir=output_dir, max_round=max_round, max_clients=max_clients)

    print("\n✓ Scalability analysis complete!")
    print(f"Output saved to: {output_dir.absolute()}")
