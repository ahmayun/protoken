import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from plotting.common import TOOL
print(f"Using TOOL name: {TOOL}")


plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # "legend.fontsize": 40,
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


def plot_gemma_all_datasets_subplots():
    results_dir_with_grad = Path("results/rq2-4/individual_layers/")
    results_dir_no_grad = Path("results/rq2-grad-disable/individual_layers/")
    output_dir = Path("paper/figures/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data with gradients
    datasets_data_with_grad = {}
    for domain_key in ["coding", "finance", "math", "medical"]:
        json_files = list(results_dir_with_grad.glob('*.json'))
        json_files = [
            f for f in json_files if domain_key in f.name and 'gemma' in f.name]

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

            datasets_data_with_grad[domain_key] = {
                'layer_indices': layer_indices,
                'accuracies': accuracies
            }

    # Load data without gradients
    datasets_data_no_grad = {}
    for domain_key in ["coding", "finance", "math", "medical"]:
        json_files = list(results_dir_no_grad.glob('*.json'))
        json_files = [
            f for f in json_files if domain_key in f.name and 'gemma' in f.name]

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

            datasets_data_no_grad[domain_key] = {
                'layer_indices': layer_indices,
                'accuracies': accuracies
            }

    fig, axes = plt.subplots(1, 4, figsize=(25, 4))

    # Adjust layout to make room for legend at top

    # Plot with uniform colors: Blue for with gradients, Black for without gradients
    color_with_grad = 'Blue'  # Blue
    color_no_grad = '#000000'    # Black

    handles = []
    labels_list = []

    avg_grads_acc = 0
    avg_no_grads_acc = 0

    for idx, (domain_key, domain_label) in enumerate(DOMAIN_NAMES.items()):
        ax = axes[idx]

        # Plot with gradients (blue solid line)
        if domain_key in datasets_data_with_grad:
            data = datasets_data_with_grad[domain_key]

            grad_accs = data['accuracies']
            middle_index = len(grad_accs) // 2
            first_half_mean = np.mean(grad_accs[:middle_index])
            last_half_mean = np.mean(grad_accs[middle_index:])
            print(f"{domain_label} - With Gradients: First half mean accuracy: {first_half_mean:.2f}%, Last half mean accuracy: {last_half_mean:.2f}%")
            print(
                f"{domain_label}- Overall Mean Accuracy with Gradients: {np.mean(grad_accs):.2f}%\n")

            avg_grads_acc += np.mean(grad_accs)

            line1 = ax.plot(data['layer_indices'], data['accuracies'],
                            marker='o', linewidth=2.5, markersize=8,
                            color=color_with_grad, linestyle='-',
                            label=f'{TOOL}-Gradients Enabled', zorder=3)[0]
            if idx == 0:
                handles.append(line1)
                labels_list.append(f'{TOOL}-Gradients Enabled')

        # Plot without gradients (black dashed line)
        if domain_key in datasets_data_no_grad:
            data = datasets_data_no_grad[domain_key]
            no_grad_accs = data['accuracies']
            middle_index = len(no_grad_accs) // 2
            first_half_mean = np.mean(no_grad_accs[:middle_index])
            last_half_mean = np.mean(no_grad_accs[middle_index:])
            print(f"{domain_label} - Without Gradients: First half mean accuracy: {first_half_mean:.2f}%, Last half mean accuracy: {last_half_mean:.2f}%")
            print(
                f"{domain_label}- Overall Mean Accuracy without Gradients: {np.mean(no_grad_accs):.2f}%\n")
            print(f"{'--' * 60}\n")
            avg_no_grads_acc += np.mean(no_grad_accs)
            line2 = ax.plot(data['layer_indices'], data['accuracies'],
                            marker='s', linewidth=2.5, markersize=8,
                            color=color_no_grad, linestyle='--',
                            label=f'{TOOL}-Gradients Disabled', zorder=3)[0]
            if idx == 0:
                handles.append(line2)
                labels_list.append(f'{TOOL}-Gradients Disabled')

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

    avg_grads_acc /= len(DOMAIN_NAMES)
    avg_no_grads_acc /= len(DOMAIN_NAMES)
    print(
        f"Average Overall Mean Accuracy with Gradients across all domains: {avg_grads_acc:.2f}%")
    print(
        f"Average Overall Mean Accuracy without Gradients across all domains: {avg_no_grads_acc:.2f}%")
    # Add legend at the top center
    fig.legend(handles, labels_list, loc='upper center', ncol=2,
               frameon=True, prop={'weight': 'bold', 'size': 20},
               bbox_to_anchor=(0.5, 1.18))

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
    plot_gemma_all_datasets_subplots()
