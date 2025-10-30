import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def generate_ablation_table(
    results_dir_with_grad="results/rq2/individual_layers",
    results_dir_no_grad="results/rq2-grad-disable/individual_layers"
):
    """
    Generate ablation table comparing gradient weighting enabled vs disabled.
    
    Args:
        results_dir_with_grad: Path to directory with gradient-enabled results
        results_dir_no_grad: Path to directory with gradient-disabled results
    
    Returns:
        pd.DataFrame: Table with columns [Domain, Model, Gradient Weighting, 
                      First Half, Last Half, Overall]
    """
    
    results_with_grad = Path(results_dir_with_grad)
    results_no_grad = Path(results_dir_no_grad)
    
    rows = []
    
    # Process each model
    for model_key, model_name in MODEL_NAMES.items():
        model_domain_data = {'enabled': {}, 'disabled': {}}
        
        # Process each domain for this model
        for domain_key, domain_name in DOMAIN_NAMES.items():
            # Find and load gradient-enabled file
            pattern_enabled = f"*{model_key}*{domain_key}*_round10.json"
            enabled_files = list(results_with_grad.glob(pattern_enabled))
            
            if not enabled_files:
                raise FileNotFoundError(f"No file found for {model_name} - {domain_name} (enabled): {pattern_enabled}")
            
            with open(enabled_files[0], 'r') as f:
                data_enabled = json.load(f)
            
            # Find and load gradient-disabled file
            pattern_disabled = f"*{model_key}*{domain_key}*_round10.json"
            disabled_files = list(results_no_grad.glob(pattern_disabled))
            
            if not disabled_files:
                raise FileNotFoundError(f"No file found for {model_name} - {domain_name} (disabled): {pattern_disabled}")
            
            with open(disabled_files[0], 'r') as f:
                data_disabled = json.load(f)
            
            # Extract accuracies for enabled (layer_X_only format)
            enabled_accuracies = []
            for layer_config in sorted(data_enabled['layer_configs_results'].keys(), 
                                      key=lambda x: int(x.replace('layer_', '').replace('_only', ''))):
                enabled_accuracies.append(data_enabled['layer_configs_results'][layer_config]['overall_accuracy'])
            
            # Extract accuracies for disabled (layer_X format)
            disabled_accuracies = []
            for layer_config in sorted(data_disabled['layer_configs_results'].keys(), 
                                      key=lambda x: int(x.replace('layer_', ''))):
                disabled_accuracies.append(data_disabled['layer_configs_results'][layer_config]['overall_accuracy'])
            
            # Calculate statistics for enabled
            midpoint_enabled = len(enabled_accuracies) // 2
            first_half_enabled = np.mean(enabled_accuracies[:midpoint_enabled])
            last_half_enabled = np.mean(enabled_accuracies[midpoint_enabled:])
            overall_enabled = np.mean(enabled_accuracies)
            
            # Calculate statistics for disabled
            midpoint_disabled = len(disabled_accuracies) // 2
            first_half_disabled = np.mean(disabled_accuracies[:midpoint_disabled])
            last_half_disabled = np.mean(disabled_accuracies[midpoint_disabled:])
            overall_disabled = np.mean(disabled_accuracies)
            
            # Store for mean calculation
            model_domain_data['enabled'][domain_key] = {
                'first_half': first_half_enabled,
                'last_half': last_half_enabled,
                'overall': overall_enabled
            }
            model_domain_data['disabled'][domain_key] = {
                'first_half': first_half_disabled,
                'last_half': last_half_disabled,
                'overall': overall_disabled
            }
            
            # Add rows for this domain
            rows.append({
                'Domain': domain_name,
                'Model': model_name,
                'Gradient Weighting': 'Enabled',
                'First Half': round(first_half_enabled, 2),
                'Last Half': round(last_half_enabled, 2),
                'Overall': round(overall_enabled, 2)
            })
            
            rows.append({
                'Domain': domain_name,
                'Model': model_name,
                'Gradient Weighting': 'Disabled',
                'First Half': round(first_half_disabled, 2),
                'Last Half': round(last_half_disabled, 2),
                'Overall': round(overall_disabled, 2)
            })
        
        # Calculate mean across domains for this model
        mean_enabled_first = np.mean([model_domain_data['enabled'][d]['first_half'] for d in DOMAIN_NAMES.keys()])
        mean_enabled_last = np.mean([model_domain_data['enabled'][d]['last_half'] for d in DOMAIN_NAMES.keys()])
        mean_enabled_overall = np.mean([model_domain_data['enabled'][d]['overall'] for d in DOMAIN_NAMES.keys()])
        
        mean_disabled_first = np.mean([model_domain_data['disabled'][d]['first_half'] for d in DOMAIN_NAMES.keys()])
        mean_disabled_last = np.mean([model_domain_data['disabled'][d]['last_half'] for d in DOMAIN_NAMES.keys()])
        mean_disabled_overall = np.mean([model_domain_data['disabled'][d]['overall'] for d in DOMAIN_NAMES.keys()])
        
        # Add mean rows
        rows.append({
            'Domain': 'Mean',
            'Model': model_name,
            'Gradient Weighting': 'Enabled',
            'First Half': round(mean_enabled_first, 2),
            'Last Half': round(mean_enabled_last, 2),
            'Overall': round(mean_enabled_overall, 2)
        })
        
        rows.append({
            'Domain': 'Mean',
            'Model': model_name,
            'Gradient Weighting': 'Disabled',
            'First Half': round(mean_disabled_first, 2),
            'Last Half': round(mean_disabled_last, 2),
            'Overall': round(mean_disabled_overall, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df


def plot_gradient_ablation_bar_chart(
    results_dir_with_grad="results/rq2/individual_layers",
    results_dir_no_grad="results/rq2-grad-disable/individual_layers",
    output_dir="paper/figures"
):
    """
    Create a grouped bar chart comparing gradient weighting enabled vs disabled
    across all model-dataset pairs.
    
    Args:
        results_dir_with_grad: Path to directory with gradient-enabled results
        results_dir_no_grad: Path to directory with gradient-disabled results
        output_dir: Directory to save the plot
    
    Returns:
        Tuple of (pdf_path, png_path)
    """
    
    # Set seaborn style and context
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    # Set matplotlib style
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })
    
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
                raise FileNotFoundError(f"No file found for {model_name} - {domain_name} (enabled)")
            
            with open(enabled_files[0], 'r') as f:
                data_enabled = json.load(f)
            
            # Find and load gradient-disabled file
            pattern_disabled = f"*{model_key}*{domain_key}*_round10.json"
            disabled_files = list(results_no_grad.glob(pattern_disabled))
            
            if not disabled_files:
                raise FileNotFoundError(f"No file found for {model_name} - {domain_name} (disabled)")
            
            with open(disabled_files[0], 'r') as f:
                data_disabled = json.load(f)
            
            # Extract accuracies for enabled (layer_X_only format)
            enabled_accs = []
            for layer_config in sorted(data_enabled['layer_configs_results'].keys(), 
                                      key=lambda x: int(x.replace('layer_', '').replace('_only', ''))):
                enabled_accs.append(data_enabled['layer_configs_results'][layer_config]['overall_accuracy'])
            
            # Extract accuracies for disabled (layer_X format)
            disabled_accs = []
            for layer_config in sorted(data_disabled['layer_configs_results'].keys(), 
                                      key=lambda x: int(x.replace('layer_', ''))):
                disabled_accs.append(data_disabled['layer_configs_results'][layer_config]['overall_accuracy'])
            
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
            'Configuration': 'Gradients Enabled'
        })
        data_rows.append({
            'Model-Dataset': label,
            'Accuracy': disabled_accuracies[i],
            'Configuration': 'Gradients Disabled'
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
    
    # Customize plot
    ax.set_xlabel('Model-Dataset Pairs', fontweight='bold', fontsize=16)
    ax.set_ylabel('Attribution Accuracy (%)', fontweight='bold', fontsize=16)
    ax.set_title('Gradient Weighting Impact on Attribution Accuracy', 
                 fontweight='bold', fontsize=18, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 105)
    
    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Customize legend
    ax.legend(
        title='Configuration',
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={'weight': 'bold', 'size': 13}
    )
    
    # Optional: Add value labels on bars
    def add_value_labels(ax):
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
    
    # Uncomment to add value labels on bars
    # add_value_labels(ax)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_path / "gradient_ablation_bar_chart.pdf"
    png_path = output_path / "gradient_ablation_bar_chart.png"
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"\nBar chart saved:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    
    plt.close(fig)
    
    return pdf_path, png_path


if __name__ == "__main__":
    # Generate the table
    df = generate_ablation_table()
    
    print("\n=== Ablation Table ===\n")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_path = Path("paper/figures/ablation_table.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nTable saved to: {output_path}")
    
    # Optionally save to LaTeX format
    latex_output = output_path.parent / "ablation_table.tex"
    df.to_latex(latex_output, index=False)
    print(f"LaTeX table saved to: {latex_output}")
    
    # Generate bar chart
    print("\n" + "="*60)
    print("Generating bar chart...")
    print("="*60)
    plot_gradient_ablation_bar_chart()
