import seaborn as sns
import matplotlib.pyplot as plt
import json


def plot_provenance_accuracy(json_path, results_dir):

    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data["metadata"]
    provenance_data = data["provenance_data"]

    if not provenance_data:
        print("No provenance data to plot")
        return

    first_round_data = next(iter(provenance_data.values()))
    dataset_names = list(first_round_data.keys())

    rounds = []
    dataset_accuracies = {dataset: [] for dataset in dataset_names}

    for round_num, round_data in provenance_data.items():
        rounds.append(int(round_num))
        for dataset in dataset_names:
            dataset_accuracies[dataset].append(round_data.get(dataset, 0))

    sns.set_theme(style="white", palette="husl")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = sns.color_palette("husl", len(dataset_names))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, dataset in enumerate(dataset_names):
        marker = markers[i % len(markers)]
        sns.lineplot(x=rounds, y=dataset_accuracies[dataset],
                     label=f'{dataset.capitalize()} Dataset', ax=ax,
                     marker=marker, markersize=8, linewidth=3, color=colors[i])

    ax.set_title(f'Provenance Accuracy Across Training Rounds\n{metadata["experiment_key"]}',
                 fontweight='bold', pad=20)
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Provenance Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    if results_dir is not None:
        plot_path = results_dir / f"provenance_{metadata['experiment_key']}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Provenance plot saved to: {plot_path}")
        plt.close()
    else:
        plt.show()
        plt.close()
