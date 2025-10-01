from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

def save_and_plot_metrics(metrics_list, results_dir, experiment_key):

    results_dir = Path(results_dir)

    json_path = results_dir / f"fl_train_metrics_{experiment_key}.json"
    
    with open(json_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)

    if not metrics_list:
        print("No metrics to plot")
        return

    df = pd.DataFrame(metrics_list)

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(data=df, x='round', y='chess_loss',
                 label='Chess', ax=ax1, marker='o')
    sns.lineplot(data=df, x='round', y='math_loss',
                 label='Math', ax=ax1, marker='s')
    sns.lineplot(data=df, x='round', y='avg_loss',
                 label='Average', ax=ax1, marker='^')
    ax1.set_title('Loss vs Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    sns.lineplot(data=df, x='round', y='chess_perplexity',
                 label='Chess', ax=ax2, marker='o')
    sns.lineplot(data=df, x='round', y='math_perplexity',
                 label='Math', ax=ax2, marker='s')
    sns.lineplot(data=df, x='round', y='avg_perplexity',
                 label='Average', ax=ax2, marker='^')
    ax2.set_title('Perplexity vs Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = results_dir / f"fl_metrics_{experiment_key}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Metrics saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")

def plot_provenance_accuracy(exp_key, results_dir):
    json_path = results_dir / f"{exp_key}_provenance.json"
    
    if not json_path.exists():
        print(f"Provenance data not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    provenance_data = data["provenance_data"]
    
    if not provenance_data:
        print("No provenance data to plot")
        return
    
    rounds = []
    chess_acc = []
    math_acc = []
    
    for round_num, round_data in provenance_data.items():
        rounds.append(int(round_num))
        chess_acc.append(round_data.get('chess', 0))
        math_acc.append(round_data.get('math', 0))
    
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = sns.color_palette("husl", 2)
    
    sns.lineplot(x=rounds, y=chess_acc, label='Chess Dataset', ax=ax, 
                marker='o', markersize=8, linewidth=3, color=colors[0])
    sns.lineplot(x=rounds, y=math_acc, label='Math Dataset', ax=ax, 
                marker='s', markersize=8, linewidth=3, color=colors[1])
    
    ax.set_title(f'Provenance Accuracy Across Training Rounds\n{metadata["experiment_key"]}', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Provenance Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    plot_path = results_dir / f"{exp_key}_provenance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Provenance plot saved to: {plot_path}")
