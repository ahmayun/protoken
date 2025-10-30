import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plotting.common import (
    MODEL_NAMES,
    DOMAIN_NAMES,
    OUTPUT_DIR,
    setup_plot_style,
    save_figure,
    TOOL,
)


def plot_gradient_ablation_bar_chart(
    output_dir,
    results_dir_with_grad="results/rq2/individual_layers",
    results_dir_no_grad="results/rq2-grad-disable/individual_layers",
):

    # Set matplotlib style
    fontsize = 20
    setup_plot_style(font_size=fontsize)

    results_with_grad = Path(results_dir_with_grad)
    results_no_grad = Path(results_dir_no_grad)

    # Collect data for plotting
    x_labels = []
    enabled_accuracies = []
    disabled_accuracies = []

    # Process each model and domain
    for model_key, model_name in MODEL_NAMES.items():
        for domain_key, domain_name in DOMAIN_NAMES.items():
            # Find and load gradient-enabled file
            pattern_enabled = f"*{model_key}*{domain_key}*_round10.json"
            enabled_files = list(results_with_grad.glob(pattern_enabled))

            if not enabled_files:
                raise FileNotFoundError(
                    f"No file found for {model_name} - {domain_name} (enabled)")

            with open(enabled_files[0], 'r') as f:
                data_enabled = json.load(f)

            # Find and load gradient-disabled file
            pattern_disabled = f"*{model_key}*{domain_key}*_round10.json"
            disabled_files = list(results_no_grad.glob(pattern_disabled))

            if not disabled_files:
                raise FileNotFoundError(
                    f"No file found for {model_name} - {domain_name} (disabled)")

            with open(disabled_files[0], 'r') as f:
                data_disabled = json.load(f)

            # Extract accuracies for enabled (layer_X_only format)
            enabled_accs = []
            for layer_config in sorted(data_enabled['layer_configs_results'].keys(),
                                       key=lambda x: int(x.replace('layer_', '').replace('_only', ''))):
                enabled_accs.append(
                    data_enabled['layer_configs_results'][layer_config]['overall_accuracy'])

            # Extract accuracies for disabled (layer_X format)
            disabled_accs = []
            for layer_config in sorted(data_disabled['layer_configs_results'].keys(),
                                       key=lambda x: int(x.replace('layer_', ''))):
                disabled_accs.append(
                    data_disabled['layer_configs_results'][layer_config]['overall_accuracy'])

            # Calculate overall average across all layers
            overall_enabled = np.mean(enabled_accs)
            overall_disabled = np.mean(disabled_accs)

            # Add to lists
            x_labels.append(f"{model_name}\n{domain_name}")
            enabled_accuracies.append(overall_enabled)
            disabled_accuracies.append(overall_disabled)

    # Prepare data in long format for seaborn
    data_rows = []
    for i, label in enumerate(x_labels):
        data_rows.append({
            'Model-Dataset': label,
            'Accuracy': enabled_accuracies[i],
            'Configuration': f'{TOOL} Gradients Enabled'
        })
        data_rows.append({
            'Model-Dataset': label,
            'Accuracy': disabled_accuracies[i],
            'Configuration': f'{TOOL} Gradients Disabled'
        })

    plot_df = pd.DataFrame(data_rows)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(18, 6))

    # Use seaborn color palette
    colors = sns.color_palette("Set2", 2)
    # Alternatively, use custom colors: colors = ['#0077B6', '#E63946']

    # Create grouped bar plot using seaborn
    bar_plot = sns.barplot(
        data=plot_df,
        x='Model-Dataset',
        y='Accuracy',
        hue='Configuration',
        palette=colors,
        ax=ax,
        edgecolor='black',
        linewidth=0.7,
        alpha=0.85
    )

    # Add hatch patterns to bars
    hatches = ['///', '+++']  # Different patterns for each configuration
    for i, container in enumerate(ax.containers):
        # Apply hatch pattern to all bars in this container
        for bar in container:
            bar.set_hatch(hatches[i % len(hatches)])

    # Customize plot
    ax.set_xlabel('LLM-Dataset Federated Learning Configuration Setting', fontweight='bold', fontsize=fontsize)
    ax.set_ylabel(f'{TOOL} Attribution\nAccuracy (%)', fontweight='bold', fontsize=fontsize)
    # ax.set_title('Gradient Weighting Impact on Attribution Accuracy',
    #              fontweight='bold', fontsize=18, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 105)

    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax.set_axisbelow(True)

    # Customize legend with patterns
    from matplotlib.patches import Patch
    legend_elements = []
    configurations = plot_df['Configuration'].unique()
    for i, config in enumerate(configurations):
        legend_elements.append(
            Patch(facecolor=colors[i], edgecolor='black', hatch=hatches[i], 
                  label=config, linewidth=0.7, alpha=0.85)
        )
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={'weight': 'bold', 'size': fontsize},
        ncol=2,
        bbox_to_anchor=(1.0, 1.1)
    )

    # Optional: Add value labels on bars
    def add_value_labels(ax):
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=fontsize/1.2, weight='bold', rotation=45)

    # Uncomment to add value labels on bars
    add_value_labels(ax)

    # Adjust layout
    plt.tight_layout()

    # Save figure using common utility
    save_figure(fig, "gradient_ablation_bar_chart",
                output_dir=Path(output_dir))

    

if __name__ == "__main__":

    print("\n" + "="*60)
    print("Generating bar chart...")
    print("="*60)
    plot_gradient_ablation_bar_chart(output_dir=OUTPUT_DIR)
