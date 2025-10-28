import json
import pathlib
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TOOL = "ProToken"

RESULTS_DIR = pathlib.Path("results/prov/backdoor")
OUTPUT_DIR = pathlib.Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "google_gemma-3-270m-it": "Gemma",
    "HuggingFaceTB_SmolLM2-360M-Instruct": "SmolLM", 
    "meta-llama_Llama-3.2-1B-Instruct": "Llama",
    "Qwen_Qwen2.5-0.5B-Instruct": "Qwen"
}

DOMAIN_NAMES = {
    "medical": "Medical",
    "math": "Math",
    'finance': "Finance",
    'coding': "Coding"
}

COLORS = {
    "attribution": "Blue",
    "token_acc": "#E63946",
    "malicious": "#DC2F02",
    "benign": "#0077B6"
}


# sns.set_style("white")
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    # 👇 Add these lines
#     "xtick.direction": "in",
#     "ytick.direction": "in",
#     "xtick.minor.visible": True,
#     "ytick.minor.visible": True,
#     "xtick.major.size": 6,
#     "xtick.minor.size": 3,
#     "ytick.major.size": 6,
#     "ytick.minor.size": 3,
# 
})


def load_json_files() -> Dict[Tuple[str, str], dict]:
    data = {}
    
    json_files = list(RESULTS_DIR.glob("provenance_*.json"))
    
    for filepath in json_files:
        filename = filepath.name
        
        model = None
        for model_key in MODEL_NAMES.keys():
            if model_key in filename:
                model = model_key
                break
        
        domain = None
        for domain_key in DOMAIN_NAMES.keys():
            if domain_key in filename:
                domain = domain_key
                break
        
        if model and domain:
            with open(filepath, 'r') as f:
                data[(model, domain)] = json.load(f)
                print(f"Loaded: {MODEL_NAMES[model]}-{DOMAIN_NAMES[domain]}")
        else:
            print(f"Warning: Could not parse {filename}")
    
    return data


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
# HELPER FUNCTIONS FOR PLOT AESTHETICS AND SAVING
# ============================================================================

def save_figure(fig, filename: str):
    """Save figure as both PDF and PNG to OUTPUT_DIR."""
    output_pdf = OUTPUT_DIR / f"{filename}.pdf"
    output_png = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"Saved {filename} to {output_pdf} and {output_png}")
    plt.close(fig)


def apply_axis_aesthetics(ax, xlabel: str = "", ylabel: str = "", 
                         row: int = 0, col: int = 0, 
                         nrows: int = 2, ncols: int = 4):
    
    if row == nrows - 1 and xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    elif xlabel:
        ax.set_xlabel("")
        
    if col == 0 and ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    elif ylabel:
        ax.set_ylabel("")
    
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', 
                  length=8, width=2, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', 
                  length=4, width=1.5, top=True, right=True)


def add_figure_legend(fig, handles, labels, ncol: int = 3):
    """Add bold legend at top center of figure with proper spacing."""
    fig.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, 1), ncol=ncol, frameon=True,
              prop={'weight': 'bold', 'size': 18})
    plt.tight_layout(rect=[0, 0, 1, 0.96])


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_tool_evaluations(data: Dict[Tuple[str, str], dict], configs: List[Tuple[str, str]]):
    fig, axes = plt.subplots(2, 8, figsize=(32, 10))
    
    all_handles = []
    all_labels = []
    
    for idx, (model, domain) in enumerate(configs):
        row = idx // 8
        col = idx % 8
        ax = axes[row, col]
        
        config_data = data[(model, domain)]
        
        rounds_attr, accuracies = extract_attribution_accuracy(config_data)
        rounds_token, benign_accs, poison_accs = extract_mean_token_accuracy(config_data)
        
        benign_accs_pct = [x * 100 for x in benign_accs]
        poison_accs_pct = [x * 100 for x in poison_accs]
        
        h1 = ax.plot(rounds_attr, accuracies, marker='o', linewidth=2.5, markersize=8,
                color=COLORS["attribution"], label=f"{TOOL} Attribution Accuracy")
        h2 = ax.plot(rounds_token, benign_accs_pct, marker='s', linewidth=2.5, markersize=8,
                linestyle='-', color='#FF9B42', label="LLM (G) - Benign Mean Token Accuracy")
        h3 = ax.plot(rounds_token, poison_accs_pct, marker='^', linewidth=2.5, markersize=8,
                linestyle='--', color=COLORS["token_acc"], label="LLM (G) - Backdoor Mean Token Accuracy")
        
        if idx == 0:
            all_handles = h1 + h2 + h3
            all_labels = [h.get_label() for h in all_handles]
        
        ax.set_ylim(10, 105)
        ax.set_title(f"{MODEL_NAMES[model]}-{DOMAIN_NAMES[domain]}", fontweight='bold')
        
        apply_axis_aesthetics(ax, xlabel="Federated Round", ylabel="Accuracy (%)",
                            row=row, col=col, nrows=2, ncols=8)
    
    add_figure_legend(fig, all_handles, all_labels, ncol=3)
    save_figure(fig, "backdoor_tool_evaluations")


def plot_confidence_boxplots(data: Dict[Tuple[str, str], dict], configs: List[Tuple[str, str]]):
    fig, axes = plt.subplots(2, 8, figsize=(32, 10))
    
     
    
    legend_handles = []
    legend_labels = []
    
    for idx, (model, domain) in enumerate(configs):
        row = idx // 8
        col = idx % 8
        ax = axes[row, col]
        
        config_data = data[(model, domain)]
        df = extract_client_contributions(config_data)
        df['probability_log'] = np.log10(df['probability'].clip(lower=1e-16))
        
        palette = {0: COLORS["malicious"], 1: COLORS["malicious"],
                  2: COLORS["benign"], 3: COLORS["benign"],
                  4: COLORS["benign"], 5: COLORS["benign"]}
        
        sns.boxplot(data=df, x='client', y='probability_log', hue='client',
                   palette=palette, ax=ax, legend=False, linewidth=1.5)
        
        ax.set_title(f"{MODEL_NAMES[model]}-{DOMAIN_NAMES[domain]}", fontweight='bold')
        
        yticks = [-15, -12, -9, -6, -3, 0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{t}}}$" for t in yticks])
        ax.set_ylim(-16, 1)
        
        apply_axis_aesthetics(ax, xlabel="Client ID", ylabel="Contribution Probability (log₁₀)",
                            row=row, col=col, nrows=2, ncols=8)
        
        if idx == 0:
            malicious_patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS["malicious"], label='Malicious (0,1)')
            benign_patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS["benign"], label='Benign (2-5)')
            legend_handles = [malicious_patch, benign_patch]
            legend_labels = ['Backdoor (Clients: 0-1)', 'Benign (Clients: 2-5)']
    
    add_figure_legend(fig, legend_handles, legend_labels, ncol=2)
    save_figure(fig, "log_probability_boxplots")


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
            'Model': MODEL_NAMES[model],
            'Domain': DOMAIN_NAMES[domain],
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


if __name__ == "__main__":
    print("Loading data...")
    data = load_json_files()
    print(f"\nLoaded {len(data)} configurations\n")

    configs = []
    for model in MODEL_NAMES.keys():
        for domain in DOMAIN_NAMES.keys():
            configs.append((model, domain))
    
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
    
    config_df.to_csv(OUTPUT_DIR / "summary_per_config.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_per_model.csv", index=False)
    print(f"\n✓ Summary statistics saved to {OUTPUT_DIR.absolute()}")
    
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    plot_tool_evaluations(data, configs)
    
    plot_confidence_boxplots(data, configs)
    
    print("\n✓ All figures generated successfully!")
    print(f"Figures saved to: {OUTPUT_DIR.absolute()}")
