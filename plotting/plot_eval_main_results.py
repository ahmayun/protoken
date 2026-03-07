import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotting.common import (
    TOOL,
    MODEL_NAMES,
    DOMAIN_NAMES,
    COLORS,
    setup_plot_style,
    save_figure,
    apply_axis_aesthetics,
)

# Short names (from reproduce.sh) -> form that appears in exp_key (sanitized model id)
MODEL_SHORT_TO_KEY = {
    "gemma": "google_gemma-3-270m-it",
    "smollm": "HuggingFaceTB_SmolLM2-360M-Instruct",
    "llama": "meta-llama_Llama-3.2-1B-Instruct",
    "qwen": "Qwen_Qwen2.5-0.5B-Instruct",
}


def _exp_key_to_filename(exp_key: str) -> str:
    """Make exp_key safe for use as a filename (no brackets)."""
    s = exp_key.replace("[", "_").replace("]", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_json_files(
    results_dir: pathlib.Path,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str], dict], Optional[str]]:
    """Load provenance JSONs from results_dir. If model_filter and dataset_filter are set,
    only load the single matching file. Returns (data, exp_key) where exp_key is the
    experiment key of the loaded run (for single-file mode) or None."""
    data = {}
    exp_key_used = None
    json_files = list(results_dir.glob("provenance_*.json"))

    if model_filter:
        model_in_key = MODEL_SHORT_TO_KEY.get(model_filter.strip().lower(), model_filter.strip().replace("/", "_"))
    else:
        model_in_key = None
    if dataset_filter:
        ds = dataset_filter.strip()
    else:
        ds = None

    for filepath in json_files:
        filename = filepath.name
        # exp_key is between "provenance_refactored_" and ".json"
        if filename.startswith("provenance_refactored_") and filename.endswith(".json"):
            exp_key = filename[len("provenance_refactored_"): -len(".json")]
        else:
            exp_key = filename

        if model_in_key and model_in_key not in filename:
            continue
        if ds and ds not in filename:
            continue

        model = None
        for model_key in MODEL_NAMES.keys():
            if model_key in filename:
                model = model_key
                break
        if model is None and model_in_key:
            model = model_in_key
        if model is None:
            continue

        domain = None
        for domain_key in DOMAIN_NAMES.keys():
            if domain_key in filename:
                domain = domain_key
                break
        if domain is None and ds:
            domain = ds
        if domain is None:
            continue

        with open(filepath, "r") as f:
            data[(model, domain)] = json.load(f)
        model_display = MODEL_NAMES.get(model, model)
        domain_display = DOMAIN_NAMES.get(domain, domain)
        print(f"Loaded: {model_display}-{domain_display}")
        exp_key_used = exp_key
        if model_filter and dataset_filter:
            break

    return data, exp_key_used


def extract_attribution_accuracy(data: dict) -> Tuple[List[int], List[float]]:
    rounds = []
    accuracies = []
    
    provenance = data["provenance"]
    for round_str, round_data in sorted(provenance.items(), key=lambda x: int(x[0])):
        rounds.append(int(round_str))
        accuracies.append(round_data["overall_accuracy"])
    
    return rounds, accuracies


def extract_mean_token_accuracy(data: dict) -> Tuple[List[int], List[float], List[float]]:
    rounds = []
    benign_accs = []
    poison_accs = []
    
    training = data["training"]
    for round_data in training:
        round_num = round_data["round"]
        
        benign_metrics = round_data["metrics_per_dataset"]["benign"]
        poison_metrics = round_data["metrics_per_dataset"]["poison"]
        
        rounds.append(round_num)
        benign_accs.append(benign_metrics["eval_mean_token_accuracy"])
        poison_accs.append(poison_metrics["eval_mean_token_accuracy"])
    
    return rounds, benign_accs, poison_accs


def extract_client_contributions(data: dict) -> pd.DataFrame:
    records = []
    
    provenance = data["provenance"]
    for round_str, round_data in provenance.items():
        for result in round_data["detailed_results"]:
            client2part = result["client2part"]
            for client, prob in client2part.items():
                records.append({
                    "round": int(round_str),
                    "client": int(client),
                    "probability": prob,
                    "type": "malicious" if int(client) in [0, 1] else "benign"
                })
    
    return pd.DataFrame(records)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_tool_evaluations(
    data: Dict[Tuple[str, str], dict],
    configs: List[Tuple[str, str]],
    output_dir: pathlib.Path,
    exp_key: Optional[str] = None,
):
    single = len(configs) == 1
    fontsize = 18 if single else 25
    setup_plot_style(font_size=fontsize)
    if single:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = np.array([[axes]])
    else:
        fig, axes = plt.subplots(2, 8, figsize=(40, 10))

    all_handles = []
    all_labels = []
    nrows, ncols = (1, 1) if single else (2, 8)

    for idx, (model, domain) in enumerate(configs):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col] if axes.ndim > 1 else axes.flat[0]

        config_data = data[(model, domain)]

        rounds_attr, accuracies = extract_attribution_accuracy(config_data)
        rounds_token, benign_accs, poison_accs = extract_mean_token_accuracy(config_data)

        benign_accs_pct = [x * 100 for x in benign_accs]
        poison_accs_pct = [x * 100 for x in poison_accs]

        h1 = ax.plot(rounds_attr, accuracies, marker="o", linewidth=2.5, markersize=8,
                color=COLORS["attribution"], label=f"{TOOL} attribution accuracy")
        h2 = ax.plot(rounds_token, benign_accs_pct, marker="s", linewidth=2.5, markersize=8,
                linestyle="-", color="#FF9B42", label="LLM (G) – benign mean token accuracy")
        h3 = ax.plot(rounds_token, poison_accs_pct, marker="^", linewidth=2.5, markersize=8,
                linestyle="--", color=COLORS["token_acc"], label="LLM (G) – backdoor mean token accuracy")

        if idx == 0:
            all_handles = h1 + h2 + h3
            all_labels = [h.get_label() for h in all_handles]

        ax.set_ylim(10, 105)
        model_display = MODEL_NAMES.get(model, model)
        domain_display = DOMAIN_NAMES.get(domain, domain)
        ax.set_title(f"{model_display}-{domain_display}", fontweight="bold", fontsize=fontsize + 2 if single else None)

        apply_axis_aesthetics(ax, xlabel="Federated round", ylabel="Accuracy (%)",
                            row=row, col=col, nrows=nrows, ncols=ncols)

    fig.legend(all_handles, all_labels, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True,
              prop={"weight": "bold", "size": fontsize})
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    basename = _exp_key_to_filename(exp_key) if exp_key else "backdoor_tool_evaluations"
    save_figure(fig, basename, output_dir=output_dir)


def plot_confidence_boxplots(
    data: Dict[Tuple[str, str], dict],
    configs: List[Tuple[str, str]],
    output_dir: pathlib.Path,
    exp_key: Optional[str] = None,
):
    fontsize = 18
    setup_plot_style(font_size=fontsize)
    single = len(configs) == 1
    if single:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = np.array([[axes]])
    else:
        fig, axes = plt.subplots(2, 8, figsize=(40, 10))

    legend_handles = []
    legend_labels = []
    nrows, ncols = (1, 1) if single else (2, 8)

    for idx, (model, domain) in enumerate(configs):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col] if axes.ndim > 1 else axes.flat[0]

        config_data = data[(model, domain)]
        df = extract_client_contributions(config_data)
        df["probability_log"] = np.log10(df["probability"].clip(lower=1e-16))

        # One box per client (0..5): build list of log-probability arrays
        client_ids = sorted(df["client"].unique())
        box_data = [df[df["client"] == c]["probability_log"].values for c in client_ids]
        positions = list(range(len(client_ids)))

        bp = ax.boxplot(
            box_data,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="gray", markersize=4),
            boxprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2, color="black"),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Color boxes: malicious (0,1) = rust/red, benign (2-5) = blue
        for i, (patch, cid) in enumerate(zip(bp["boxes"], client_ids)):
            patch.set_facecolor(COLORS["malicious"] if cid in (0, 1) else COLORS["benign"])
            patch.set_edgecolor("black")
            patch.set_linewidth(1)

        model_display = MODEL_NAMES.get(model, model)
        domain_display = DOMAIN_NAMES.get(domain, domain)
        ax.set_title(f"{model_display}-{domain_display}", fontweight="bold", fontsize=fontsize + 2)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(c) for c in client_ids])

        yticks = [-15, -12, -9, -6, -3, 0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{t}}}$" for t in yticks])
        ax.set_ylim(-16, 1)

        apply_axis_aesthetics(ax, xlabel="Client ID", ylabel="Contribution probability (log)",
                            row=row, col=col, nrows=nrows, ncols=ncols)

        if idx == 0:
            malicious_patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS["malicious"], label="Malicious (0,1)")
            benign_patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS["benign"], label="Benign (2-5)")
            legend_handles = [malicious_patch, benign_patch]
            legend_labels = ["Responsible clients (0-1)", "Non-responsible clients (2-5)"]

    fig.legend(legend_handles, legend_labels, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True,
              prop={"weight": "bold", "size": fontsize})
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    basename = _exp_key_to_filename(exp_key) + "_boxplots" if exp_key else "log_probability_boxplots"
    save_figure(fig, basename, output_dir=output_dir)


def compute_summary_statistics(data: Dict[Tuple[str, str], dict], configs: List[Tuple[str, str]]):
    config_rows = []
    
    for model, domain in configs:
        config_data = data[(model, domain)]
        
        rounds, accuracies = extract_attribution_accuracy(config_data)
        
        df = extract_client_contributions(config_data)
        mal_probs = df[df['type'] == 'malicious']['probability']
        ben_probs = df[df['type'] == 'benign']['probability']
        
        mal_mean = mal_probs.mean()
        ben_mean = ben_probs.mean()
        separation = np.log10(mal_mean / ben_mean) if ben_mean > 0 else 0
        
        config_rows.append({
            'Model': MODEL_NAMES.get(model, model),
            'Domain': DOMAIN_NAMES.get(domain, domain),
            'Mean_Acc': np.mean(accuracies),
            'Min_Acc': np.min(accuracies),
            'Max_Acc': np.max(accuracies),
            'Configs_100': 1 if np.max(accuracies) >= 99.5 else 0,
            'Mal_Prob': mal_mean,
            'Ben_Prob': ben_mean,
            'Separation': separation
        })
    
    config_df = pd.DataFrame(config_rows)
    
    model_summary = []
    for model_key, model_name in MODEL_NAMES.items():
        model_configs = config_df[config_df['Model'] == model_name]
        model_summary.append({
            'Model': model_name,
            'Mean_Acc': model_configs['Mean_Acc'].mean(),
            'Mal_Prob': model_configs['Mal_Prob'].mean(),
            'Ben_Prob': model_configs['Ben_Prob'].mean(),
            'Separation': model_configs['Separation'].mean()
        })
    
    model_df = pd.DataFrame(model_summary)
    
    overall = {
        'Model': 'Overall',
        'Mean_Acc': config_df['Mean_Acc'].mean(),
        'Mal_Prob': config_df['Mal_Prob'].mean(),
        'Ben_Prob': config_df['Ben_Prob'].mean(),
        'Separation': config_df['Separation'].mean()
    }
    
    overall_df = pd.DataFrame([overall])
    summary_df = pd.concat([model_df, overall_df], ignore_index=True)
    
    return config_df, summary_df


def _parse_args():
    p = argparse.ArgumentParser(description="Plot main eval results (Fig 2 & 3) from provenance JSONs.")
    p.add_argument("--model", default=None, help="Model short name (gemma|smollm|llama|qwen) or key. Plot only this model.")
    p.add_argument("--dataset", default=None, help="Dataset name (e.g. medical). Plot only this dataset.")
    p.add_argument("--results_dir", required=True, help="Directory containing provenance_*.json files.")
    p.add_argument("--output_dir", required=True, help="Directory for output PNG/PDF and CSVs.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data, exp_key = load_json_files(
        results_dir,
        model_filter=args.model,
        dataset_filter=args.dataset,
    )
    if not data:
        print("No matching provenance JSONs found. Exiting.")
        raise SystemExit(1)
    print(f"\nLoaded {len(data)} configuration(s)\n")

    configs = list(data.keys())

    print("="*80)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*80)

    config_df, summary_df = compute_summary_statistics(data, configs)

    print("\n" + "="*80)
    print("PAPER-READY SUMMARY (Attribution Accuracy Only)")
    print("="*80)

    overall_mean = config_df['Mean_Acc'].mean()
    overall_min = config_df['Min_Acc'].min()
    overall_max = config_df['Max_Acc'].max()

    print(f"\nOverall: {overall_mean:.2f}% (range: {overall_min:.1f}% - {overall_max:.1f}%)")
    print(f"\nPer-Model:")
    for _, row in summary_df.iterrows():
        if row['Model'] != 'Overall':
            print(f"  {row['Model']:8s} {row['Mean_Acc']:6.2f}%")

    print("\n" + "="*80)
    print("DETAILED STATISTICS (For Reference)")
    print("="*80)
    pd.options.display.float_format = '{:.2f}'.format
    print(summary_df.to_string(index=False))

    print("\n" + "="*80)
    print("PER-CONFIGURATION DETAILS (For Reference)")
    print("="*80)
    print(config_df.to_string(index=False))

    config_df.to_csv(output_dir / "summary_per_config.csv", index=False)
    summary_df.to_csv(output_dir / "summary_per_model.csv", index=False)
    print(f"\n✓ Summary statistics saved to {output_dir.absolute()}")

    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    plot_tool_evaluations(data, configs, output_dir=output_dir, exp_key=exp_key)
    plot_confidence_boxplots(data, configs, output_dir=output_dir, exp_key=exp_key)

    print("\n✓ All figures generated successfully!")
    print(f"Figures saved to: {output_dir.absolute()}")
