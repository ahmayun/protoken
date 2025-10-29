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

DOMAIN_NAMES = {
    "medical": "Medical",
    "math": "Math",
    "finance": "Finance",
    "coding": "Coding"
}


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


def plot_gemma_all_datasets():
    results_dir = Path("results/rq2-4/individual_layers/")
    output_dir = Path("paper/figures/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_data = {}
    
    for domain_key in ["coding", "finance", "math", "medical"]:
        pattern = f"*google_gemma-3-270m-it*Datasets-['{domain_key}']*round10.json"
        json_files = list(results_dir.glob('*.json'))

        # for f in json_files:
        #     print(f"Found JSON file for domain '{domain_key}': {f}")
        
        # _ = input("Press Enter to continue...")
        json_files = [f for f in json_files if domain_key in f.name and 'gemma' in f.name]
        
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            layer_indices = []
            accuracies = []
            for config_name in sorted(data['layer_configs_results'].keys(), 
                                     key=lambda x: int(x.replace('layer_', '').replace('_only', ''))):
                config_data = data['layer_configs_results'][config_name]
                layer_indices.append(config_data['layer_index'])
                accuracies.append(config_data['overall_accuracy'])
            
            datasets_data[domain_key] = {
                'layer_indices': layer_indices,
                'accuracies': accuracies
            }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#0077B6', '#E63946', '#06A77D', '#F77F00']
    markers = ['o', 's', '^', 'D']
    
    for idx, (domain_key, domain_label) in enumerate(DOMAIN_NAMES.items()):
        if domain_key in datasets_data:
            data = datasets_data[domain_key]
            ax.plot(data['layer_indices'], data['accuracies'],
                   marker=markers[idx], linewidth=2.5, markersize=8,
                   color=colors[idx], label=domain_label, zorder=3)
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Attribution Accuracy (%)', fontweight='bold')
    ax.set_title('Gemma: Per-Layer Attribution Accuracy', fontweight='bold', pad=20)
    
    ax.set_ylim(-5, 110)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in',
                   length=8, width=2, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=4, width=1.5, top=True, right=True)
    
    ax.legend(loc='best', frameon=True, prop={'weight': 'bold', 'size': 14})
    
    plt.tight_layout()
    
    output_pdf = output_dir / "gemma_all_datasets_comparison.pdf"
    output_png = output_dir / "gemma_all_datasets_comparison.png"
    
    fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', dpi=300)
    
    print(f"Saved plots:")
    print(f"  PDF: {output_pdf}")
    print(f"  PNG: {output_png}")
    
    plt.close(fig)
    
    return output_pdf, output_png


def plot_gemma_all_datasets_subplots():
    results_dir = Path("results/rq2-4/individual_layers/")
    output_dir = Path("paper/figures/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_data = {}
    
    for domain_key in ["coding", "finance", "math", "medical"]:
        json_files = list(results_dir.glob('*.json'))
        json_files = [f for f in json_files if domain_key in f.name and 'gemma' in f.name]
        
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            layer_indices = []
            accuracies = []
            for config_name in sorted(data['layer_configs_results'].keys(), 
                                     key=lambda x: int(x.replace('layer_', '').replace('_only', ''))):
                config_data = data['layer_configs_results'][config_name]
                layer_indices.append(config_data['layer_index'])
                accuracies.append(config_data['overall_accuracy'])
            
            datasets_data[domain_key] = {
                'layer_indices': layer_indices,
                'accuracies': accuracies
            }
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    colors = ['#0077B6', '#E63946', '#06A77D', '#F77F00']
    
    for idx, (domain_key, domain_label) in enumerate(DOMAIN_NAMES.items()):
        if domain_key in datasets_data:
            ax = axes[idx]
            data = datasets_data[domain_key]
            
            ax.plot(data['layer_indices'], data['accuracies'],
                   marker='o', linewidth=2.5, markersize=8,
                   color=colors[idx], zorder=3)
            
            ax.set_title(domain_label, fontweight='bold', pad=10)
            ax.set_xlabel('Layer Index', fontweight='bold')
            
            if idx == 0:
                ax.set_ylabel('Attribution Accuracy (%)', fontweight='bold')
            
            ax.set_ylim(-5, 110)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)
            ax.set_axisbelow(True)
            
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', direction='in',
                          length=8, width=2, top=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in',
                          length=4, width=1.5, top=True, right=True)
    
    plt.tight_layout()
    
    output_pdf = output_dir / "gemma_all_datasets_subplots.pdf"
    output_png = output_dir / "gemma_all_datasets_subplots.png"
    
    fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', dpi=300)
    
    print(f"Saved plots:")
    print(f"  PDF: {output_pdf}")
    print(f"  PNG: {output_png}")
    
    plt.close(fig)
    
    return output_pdf, output_png


if __name__ == "__main__":
    plot_gemma_all_datasets()
    plot_gemma_all_datasets_subplots()
