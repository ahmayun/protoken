import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
})


def plot_individual_layer_accuracy(json_path, output_dir=None, layer_ranges=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    exp_key = data['experiment_key']
    round_num = data['round_num']
    layer_configs = data['layer_configs_results']
    
    layer_indices = []
    accuracies = []
    for config_name in sorted(layer_configs.keys(), key=lambda x: int(x.replace('layer_', '').replace('_only', ''))):
        config_data = layer_configs[config_name]
        layer_indices.append(config_data['layer_index'])
        accuracies.append(config_data['overall_accuracy'])
    
    if layer_ranges is None:
        total_layers = max(layer_indices) + 1
        early_end = total_layers // 3
        middle_end = 2 * total_layers // 3
        layer_ranges = {
            'early': (0, early_end),
            'middle': (early_end, middle_end),
            'late': (middle_end, total_layers)
        }
    
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'early': '#FFCCCB',
        'middle': '#FFF4C4', 
        'late': '#C8E6C9'
    }
    
    ax.axvspan(layer_ranges['early'][0], layer_ranges['early'][1] - 0.5, 
               alpha=0.2, color=colors['early'], zorder=0)
    ax.axvspan(layer_ranges['middle'][0], layer_ranges['middle'][1] - 0.5,
               alpha=0.2, color=colors['middle'], zorder=0)
    ax.axvspan(layer_ranges['late'][0], layer_ranges['late'][1],
               alpha=0.2, color=colors['late'], zorder=0)
    
    ax.plot(layer_indices, accuracies, 
            marker='o', linewidth=2.5, markersize=10,
            color='#0077B6', label='Attribution Accuracy', zorder=3)
    
    early_accs = [acc for idx, acc in zip(layer_indices, accuracies) 
                  if layer_ranges['early'][0] <= idx < layer_ranges['early'][1]]
    middle_accs = [acc for idx, acc in zip(layer_indices, accuracies)
                   if layer_ranges['middle'][0] <= idx < layer_ranges['middle'][1]]
    late_accs = [acc for idx, acc in zip(layer_indices, accuracies)
                 if layer_ranges['late'][0] <= idx < layer_ranges['late'][1]]
    
    early_mean = np.mean(early_accs) if early_accs else 0
    middle_mean = np.mean(middle_accs) if middle_accs else 0
    late_mean = np.mean(late_accs) if late_accs else 0
    
    early_center = (layer_ranges['early'][0] + layer_ranges['early'][1]) / 2
    middle_center = (layer_ranges['middle'][0] + layer_ranges['middle'][1]) / 2
    late_center = (layer_ranges['late'][0] + layer_ranges['late'][1]) / 2
    
    y_pos = ax.get_ylim()[1] * 0.95
    
    ax.text(early_center, y_pos, f'Early\n{early_mean:.1f}%',
            ha='center', va='top', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.text(middle_center, y_pos, f'Middle\n{middle_mean:.1f}%',
            ha='center', va='top', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.text(late_center, y_pos, f'Late\n{late_mean:.1f}%',
            ha='center', va='top', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Attribution Accuracy (%)', fontweight='bold')
    ax.set_title('Per-Layer Provenance Attribution Accuracy', fontweight='bold', pad=20)
    
    ax.set_xlim(-0.5, max(layer_indices) + 0.5)
    ax.set_ylim(-5, 110)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', 
                   length=8, width=2, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=4, width=1.5, top=True, right=True)
    
    plt.tight_layout()
    
    output_pdf = output_dir / f"{exp_key}_individual_layer_accuracy_round{round_num}.pdf"
    output_png = output_dir / f"{exp_key}_individual_layer_accuracy_round{round_num}.png"
    
    fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', dpi=300)
    
    print(f"Saved plots:")
    print(f"  PDF: {output_pdf}")
    print(f"  PNG: {output_png}")
    
    plt.close(fig)
    
    return output_pdf, output_png


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot RQ2 Individual Layer Accuracy')
    parser.add_argument('--json', type=str, 
                       default='results/rq2/individual_layers/all_experiments_summary_round10.json',
                       help='Path to JSON file with layer accuracy data')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to JSON directory)')
    parser.add_argument('--exp_key', type=str, default=None,
                       help='Experiment key to plot (if not provided, uses first experiment)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RQ2: Plotting Individual Layer Attribution Accuracy")
    print("="*80)
    
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        exit(1)
    
    print(f"\nLoading data from: {json_path}")
    
    output_dir = args.output_dir if args.output_dir else json_path.parent
    
    plot_individual_layer_accuracy(json_path, output_dir)
    
    print("\n" + "="*80)
    print("Plot generation complete!")
    print("="*80)
