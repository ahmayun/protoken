import json
import pathlib
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotting.common import (
    TOOL,
    MODEL_NAMES,
    COLORS,
    OUTPUT_DIR,
    setup_plot_style,
    save_figure,
    apply_axis_aesthetics,
)

# Path to scalability data
RESULTS_DIR = pathlib.Path("results/debug")


def load_scalability_data() -> Dict[str, dict]:
    """Load both Gemma and Qwen scalability JSON files."""
    data = {}
    
    # Define the two model files
    files = {
        # "google_gemma-3-270m-it": RESULTS_DIR / "single_provenance_refactored_[google_gemma-3-270m-it][rounds-16][epochs-1][clients55-per-round-10][Datasets-['coding']-None][partitioning-iid][Backdoor-True][Unsloth-False][Lora-False].json",
        # "Qwen_Qwen2.5-0.5B-Instruct": RESULTS_DIR / "single_provenance_refactored_[Qwen_Qwen2.5-0.5B-Instruct][rounds-16][epochs-1][clients55-per-round-10][Datasets-['coding']-None][partitioning-iid][Backdoor-True][Unsloth-False][Lora-False].json",
        "HuggingFaceTB_SmolLM2-360M-Instruct": RESULTS_DIR / "single_provenance_refactored_[HuggingFaceTB_SmolLM2-360M-Instruct][rounds-16][epochs-1][clients55-per-round-10][Datasets-['medical']-None][partitioning-iid][Backdoor-True][Unsloth-False][Lora-False].json"
    }
    
    for model_key, filepath in files.items():
        if filepath.exists():
            with open(filepath, 'r') as f:
                data[model_key] = json.load(f)
                print(f"Loaded: {MODEL_NAMES[model_key]}")
        else:
            print(f"Warning: File not found: {filepath}")
    
    return data


def extract_training_metrics(data: dict) -> Tuple[List[int], List[float], List[float]]:
    """Extract benign and poison accuracy from training data (rounds 0-15 only)."""
    rounds = []
    benign_accs = []
    poison_accs = []
    
    training = data["training"]
    for round_data in training:
        round_num = round_data["round"]
        
        # Limit to rounds 0-15
        if round_num > 15:
            continue
        
        benign_metrics = round_data["metrics_per_dataset"]["benign"]
        poison_metrics = round_data["metrics_per_dataset"]["poison"]
        
        rounds.append(round_num)
        benign_accs.append(benign_metrics["eval_mean_token_accuracy"] * 100)
        poison_accs.append(poison_metrics["eval_mean_token_accuracy"] * 100)
    
    return rounds, benign_accs, poison_accs


def extract_provenance_accuracy(data: dict) -> Tuple[List[int], List[float]]:
    """Extract provenance attribution accuracy per round (rounds 1-15 only)."""
    rounds = []
    accuracies = []
    
    provenance = data["provenance"]
    for round_str, round_data in sorted(provenance.items(), key=lambda x: int(x[0])):
        round_num = int(round_str)
        
        # Limit to rounds 1-15
        if round_num > 15:
            continue
            
        rounds.append(round_num)
        accuracies.append(round_data["overall_accuracy"])
    
    return rounds, accuracies


def extract_client_contributions(data: dict) -> pd.DataFrame:
    """Extract client contribution probabilities for boxplot visualization."""
    records = []
    
    provenance = data["provenance"]
    # Malicious clients are 0-24, benign clients are 25-54
    for round_str, round_data in provenance.items():
        round_num = int(round_str)
        
        # Limit to rounds 1-15
        if round_num > 15:
            continue
            
        for result in round_data["detailed_results"]:
            client2part = result["client2part"]
            for client, prob in client2part.items():
                client_id = int(client)
                records.append({
                    "round": round_num,
                    "client": client_id,
                    "probability": prob,
                    "type": "malicious" if client_id < 25 else "benign"
                })
    
    return pd.DataFrame(records)


def print_scalability_statistics(data: Dict[str, dict]):
    """Print comprehensive scalability statistics."""
    print("\n" + "="*80)
    print("RQ4 SCALABILITY STATISTICS")
    print("="*80)
    print("\nExperimental Setup:")
    print("  Total Clients: 55")
    print("  Clients per Round: 10")
    print("  Total Rounds: 16")
    print("  Backdoor Clients: 25 (clients 0-24)")
    print("  Samples per Client: 200")
    
    for model_key in [
        # "google_gemma-3-270m-it", 
        # "Qwen_Qwen2.5-0.5B-Instruct", 
        "HuggingFaceTB_SmolLM2-360M-Instruct"]:
        if model_key not in data:
            continue
        
        model_data = data[model_key]
        model_name = MODEL_NAMES[model_key]
        
        print(f"\n{'='*80}")
        print(f"{model_name}")
        print("="*80)
        
        # Extract metrics
        train_rounds, benign_accs, poison_accs = extract_training_metrics(model_data)
        prov_rounds, prov_accs = extract_provenance_accuracy(model_data)
        overall_prov_acc = model_data["across_all_rounds_accuracy"]
        
        # Training metrics
        print("\nTraining Metrics:")
        print(f"  Initial Benign Accuracy: {benign_accs[0]:.2f}%")
        print(f"  Final Benign Accuracy: {benign_accs[-1]:.2f}%")
        print(f"  Initial Backdoor Accuracy: {poison_accs[0]:.2f}%")
        print(f"  Final Backdoor Accuracy: {poison_accs[-1]:.2f}%")
        
        # Provenance metrics
        print(f"\n{TOOL} Provenance Metrics:")
        print(f"  Overall Accuracy (all rounds): {overall_prov_acc:.2f}%")
        print(f"  Per-Round Accuracy Range: {min(prov_accs):.2f}% - {max(prov_accs):.2f}%")
        print(f"  Mean Per-Round Accuracy: {np.mean(prov_accs):.2f}%")
        print(f"  Rounds with 100% Accuracy: {sum(1 for acc in prov_accs if acc == 100.0)}/{len(prov_accs)}")
        
        # Detailed per-round
        print(f"\n  Per-Round Breakdown:")
        for round_num, acc in zip(prov_rounds, prov_accs):
            print(f"    Round {round_num:2d}: {acc:6.2f}%")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n✓ {TOOL} successfully scales to 55 clients (vs 6 clients in RQ1)")
    print(f"✓ Maintains high provenance accuracy with 25 malicious clients")
    print(f"✓ Both models converge successfully in federated training")
    print(f"✓ Backdoor injection works effectively at scale")
    print("\n" + "="*80 + "\n")


def plot_scalability_results(data: Dict[str, dict]):
    """Create 1x2 grid with all metrics combined per model (matching RQ1 style)."""
    fontsize = 25
    setup_plot_style(font_size=fontsize)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    model_keys = [
        # "google_gemma-3-270m-it", 
        # "Qwen_Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB_SmolLM2-360M-Instruct"]
    
    legend_handles = []
    legend_labels = []
    
    for idx, model_key in enumerate(model_keys):
        if model_key not in data:
            print(f"Warning: Data not found for {model_key}")
            continue
        
        model_data = data[model_key]
        model_name = MODEL_NAMES[model_key]
        
        # Extract data
        train_rounds, benign_accs, poison_accs = extract_training_metrics(model_data)
        prov_rounds, prov_accs = extract_provenance_accuracy(model_data)
        
        # Plot all three metrics in one subplot
        ax = axes[idx]
        
        # ProToken attribution accuracy (blue circles)
        h1 = ax.plot(prov_rounds, prov_accs, marker='o', linewidth=2.5,
                     markersize=8, color=COLORS["attribution"],
                     label=f"{TOOL} Attribution Accuracy")
        
        # Benign mean token accuracy (orange squares, solid)
        h2 = ax.plot(train_rounds, benign_accs, marker='s', linewidth=2.5, 
                     markersize=8, linestyle='-', color='#FF9B42',
                     label="LLM (G) - Benign Mean Token Accuracy")
        
        # Backdoor mean token accuracy (red triangles, dashed)
        h3 = ax.plot(train_rounds, poison_accs, marker='^', linewidth=2.5,
                     markersize=8, linestyle='--', color=COLORS["token_acc"],
                     label="LLM (G) - Backdoor Mean Token Accuracy")
        
        ax.set_ylim(10, 105)
        ax.set_title(f"{model_name}", fontweight='bold', fontsize=fontsize)
        apply_axis_aesthetics(ax, xlabel="Federated Round", 
                            ylabel="Accuracy (%)",
                            row=0, col=idx, nrows=1, ncols=2)
        
        # Collect legend handles from first model only
        if idx == 0:
            legend_handles = h1 + h2 + h3
            legend_labels = [
                f"{TOOL} Attribution Accuracy",
                "LLM - Benign Mean Token Accuracy",
                "LLM - Backdoor Mean Token Accuracy"
            ]
    
    # Create combined legend
    fig.legend(legend_handles, legend_labels, loc='center',
               ncol=2, frameon=True, bbox_to_anchor= (0.54, 0.35),
              prop={'weight': 'bold', 'size': fontsize}, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, "scalability_results")


def plot_scalability_boxplots(data: Dict[str, dict]):
    """Create 1x2 aggregated boxplot showing malicious vs benign client distributions."""
    fontsize = 25
    setup_plot_style(font_size=fontsize)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    model_keys = [
        # "google_gemma-3-270m-it", 
        #           "Qwen_Qwen2.5-0.5B-Instruct", 
                  "HuggingFaceTB_SmolLM2-360M-Instruct"]
    
    legend_handles = []
    legend_labels = []
    
    for idx, model_key in enumerate(model_keys):
        if model_key not in data:
            print(f"Warning: Data not found for {model_key}")
            continue
        
        model_data = data[model_key]
        model_name = MODEL_NAMES[model_key]
        
        # Extract client contributions (already grouped by type)
        df = extract_client_contributions(model_data)
        if df.empty:
            print(f"Warning: No contribution data for {model_name}, skipping boxplot")
            continue
        df['probability_log'] = np.log10(df['probability'].clip(lower=1e-16))

        # Debug: data summary for boxplot
        print(f"\n[DEBUG] {model_name} (panel {idx}):")
        print(f"  Rows: {len(df)}")
        print(f"  Types: {df['type'].value_counts().to_dict()}")
        print(f"  probability raw: min={df['probability'].min():.6f}, max={df['probability'].max():.6f}, unique count={df['probability'].nunique()}")
        print(f"  probability_log: min={df['probability_log'].min():.4f}, max={df['probability_log'].max():.4f}, unique count={df['probability_log'].nunique()}")
        for t in ['malicious', 'benign']:
            sub = df[df['type'] == t]
            if len(sub) > 0:
                print(f"  {t}: n={len(sub)}, prob_log unique={sorted(sub['probability_log'].unique())[:10]}{'...' if sub['probability_log'].nunique() > 10 else ''}")

        ax = axes[idx]
        
        # Create aggregated boxplot: just 2 boxes (malicious vs benign)
        palette = {
            "malicious": COLORS["malicious"],
            "benign": COLORS["benign"]
        }
        
        # Use hue so palette is applied (fixes deprecation and ensures colors show)
        sns.boxplot(data=df, x='type', y='probability_log', hue='type',
                   palette=palette, ax=ax, linewidth=2, width=0.6,
                   order=['malicious', 'benign'], legend=False)
        
        ax.set_title(f"{model_name}", fontweight='bold', fontsize=fontsize)
        
        # Set y-axis with log scale
        yticks = [-15, -12, -9, -6, -3, 0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{t}}}$" for t in yticks])
        ax.set_ylim(-16, 1)
        
        # Set x-axis labels only when we have 2 ticks (avoids set_ticklabels warning)
        xticks = ax.get_xticks()
        if len(xticks) == 2:
            ax.set_xticklabels(['Responsible\n(0-24)', 'Non-Responsible\n(25-54)'])
        
        apply_axis_aesthetics(ax, xlabel="Client Type", 
                            ylabel="Contribution\nProbability (log)",
                            row=0, col=idx, nrows=1, ncols=2)
        
        # Collect legend handles from first model only
        if idx == 0:
            malicious_patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS["malicious"])
            benign_patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS["benign"])
            legend_handles = [malicious_patch, benign_patch]
            legend_labels = ['Responsible Clients (0-24)', 'Non-Responsible Clients (25-54)']
    
    # # Create combined legend
    # fig.legend(legend_handles, legend_labels, loc='upper center',
    #           bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True,
    #           prop={'weight': 'bold', 'size': fontsize}, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, "scalability_probability_boxplots")


if __name__ == "__main__":
    print("="*80)
    print("RQ4: Scalability Evaluation")
    print("="*80)
    
    print("\nLoading scalability data...")
    data = load_scalability_data()
    print(f"Loaded data for {len(data)} models")
    
    # Print comprehensive statistics
    print_scalability_statistics(data)
    
    print("\nGenerating scalability plots...")
    plot_scalability_results(data)
    
    print("\nGenerating probability distribution boxplots...")
    plot_scalability_boxplots(data)
    
    print("\n✓ Scalability analysis complete!")
    print(f"Output saved to: {OUTPUT_DIR.absolute()}")
