from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

def save_and_plot_metrics(metrics_list, results_dir):

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "fl_metrics.json"
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

    plot_path = results_dir / "fl_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Metrics saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")