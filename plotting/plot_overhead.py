import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from plotting.common import (
    TOOL,
    MODEL_NAMES,
    COLORS,
    OUTPUT_DIR,
    setup_plot_style,
    save_figure,
    apply_axis_aesthetics,
)

# Path to the overhead data
DATA_FILE = pathlib.Path("results/rq3-overhead/overhead_analysis/summary_round10.json")


def load_overhead_data():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)


def extract_model_data(config_data):
   
    results = config_data["results"]
    
    num_layers = []
    overhead_mean = []
    overhead_std = []
    accuracy = []
    
    # Sort by number of layers
    sorted_keys = sorted(results.keys(), key=lambda x: results[x]["num_layers"])
    
    for key in sorted_keys:
        data = results[key]
        num_layers.append(data["num_layers"])
        overhead_mean.append(data["prov_mean"])
        overhead_std.append(data["prov_std"])
        accuracy.append(data["accuracy"])
    
    return num_layers, overhead_mean, overhead_std, accuracy


def print_overhead_statistics(data):
    """
    Print comprehensive overhead statistics to verify numbers in RQ3 section.
    Computes all metrics mentioned in the paper.
    """
    print("\n" + "="*80)
    print("RQ3 OVERHEAD STATISTICS")
    print("="*80)
    
    # Model keys in order
    model_keys = [
        "google_gemma-3-270m-it",
        "Qwen_Qwen2.5-0.5B-Instruct", 
        "HuggingFaceTB_SmolLM2-360M-Instruct",
        "meta-llama_Llama-3.2-1B-Instruct"
    ]
    
    for model_key in model_keys:
        # Find the config for this model in the data
        config_key = None
        for key in data.keys():
            if model_key in key:
                config_key = key
                break
        
        if not config_key:
            print(f"\nWarning: Could not find data for {model_key}")
            continue
        
        config_data = data[config_key]
        model_name = MODEL_NAMES.get(model_key, model_key)
        total_layers = config_data["total_layers"]
        results = config_data["results"]
        
        # Extract min and max configurations
        sorted_keys = sorted(results.keys(), key=lambda x: results[x]["num_layers"])
        min_config = results[sorted_keys[0]]
        max_config = results[sorted_keys[-1]]
        
        min_layers = min_config["num_layers"]
        max_layers = max_config["num_layers"]
        min_overhead = min_config["prov_mean"]
        max_overhead = max_config["prov_mean"]
        
        # Calculate percentage increases
        overhead_increase_pct = ((max_overhead - min_overhead) / min_overhead) * 100
        layer_increase_pct = ((max_layers - min_layers) / min_layers) * 100
        
        # Check accuracy
        all_accuracy = [results[k]["accuracy"] for k in results.keys()]
        accuracy_check = "✓" if all(acc == 100.0 for acc in all_accuracy) else "✗"
        
        # Print formatted statistics
        print(f"\n{model_name} ({total_layers} total layers):")
        print(f"  Overhead Range:")
        print(f"    Min: {min_overhead:.2f}s with {min_layers} layers")
        print(f"    Max: {max_overhead:.2f}s with {max_layers} layers")
        print(f"  Overhead Increase: {overhead_increase_pct:.0f}% for {layer_increase_pct:.0f}% layer increase")
        print(f"  Accuracy: 100% across all {len(results)} configurations {accuracy_check}")
        
        # Print detailed breakdown
        print(f"  Detailed configurations:")
        for config_name in sorted_keys:
            config = results[config_name]
            print(f"    {config['num_layers']:2d} layers: {config['prov_mean']:.2f}s (±{config['prov_std']:.3f}), Acc: {config['accuracy']:.0f}%")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print("="*80)
    
    all_overheads = []
    all_accuracies = []
    min_overhead_overall = float('inf')
    max_overhead_overall = 0
    
    for model_key in model_keys:
        config_key = None
        for key in data.keys():
            if model_key in key:
                config_key = key
                break
        
        if config_key:
            results = data[config_key]["results"]
            for config in results.values():
                all_overheads.append(config["prov_mean"])
                all_accuracies.append(config["accuracy"])
                min_overhead_overall = min(min_overhead_overall, config["prov_mean"])
                max_overhead_overall = max(max_overhead_overall, config["prov_mean"])
    
    print(f"\nOverhead across all models and configurations:")
    print(f"  Range: {min_overhead_overall:.2f}s - {max_overhead_overall:.2f}s")
    print(f"  Mean: {np.mean(all_overheads):.2f}s")
    print(f"  Std: {np.std(all_overheads):.2f}s")
    
    print(f"\nAccuracy verification:")
    print(f"  All configurations achieve 100% accuracy: {all(acc == 100.0 for acc in all_accuracies)}")
    print(f"  Total configurations tested: {len(all_overheads)}")
    
    print("\n" + "="*80 + "\n")


def plot_dual_axis_overhead(data):
    """Create a 2x2 grid of dual y-axis plots for overhead vs accuracy."""
    
    fontsize = 25
    setup_plot_style(font_size=fontsize)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define model order and positions
    model_keys = [
        "google_gemma-3-270m-it",      # Top-left
        "Qwen_Qwen2.5-0.5B-Instruct",  # Top-right
        "HuggingFaceTB_SmolLM2-360M-Instruct",  # Bottom-left
        "meta-llama_Llama-3.2-1B-Instruct"      # Bottom-right
    ]
    
    all_overhead_handles = []
    all_accuracy_handles = []
    
    for idx, model_key in enumerate(model_keys):
        # Find the config for this model in the data
        config_key = None
        for key in data.keys():
            if model_key in key:
                config_key = key
                break
        
        if not config_key:
            print(f"Warning: Could not find data for {model_key}")
            continue
        
        config_data = data[config_key]
        num_layers, overhead_mean, overhead_std, accuracy = extract_model_data(config_data)
        
        # Get the subplot
        ax1 = axes[idx]
        row = idx // 2
        col = idx % 2
        
        # Plot overhead on left y-axis (primary)
        color_overhead = 'Blue'  # Blue
        h1 = ax1.errorbar(num_layers, overhead_mean, yerr=overhead_std,
                          marker='o', linewidth=2.5, markersize=8,
                          color=color_overhead, capsize=5, capthick=2,
                          label='Provenance Time', zorder=3)
        ax1.set_xlabel('Number of Layers', fontweight='bold')
        ax1.set_ylabel('Avg. Provenance Time (s)', fontweight='bold', color=color_overhead)
        ax1.tick_params(axis='y', labelcolor=color_overhead)
        
        # Create second y-axis for accuracy
        ax2 = ax1.twinx()
        color_accuracy = 'k'  # Red
        h2 = ax2.plot(num_layers, accuracy, marker='s', linewidth=2.5,
                      markersize=8, color=color_accuracy,
                      label=f'{TOOL} Accuracy', linestyle='--', zorder=3)
        ax2.set_ylabel(f'{TOOL} Accuracy (%)', fontweight='bold', color=color_accuracy)
        ax2.tick_params(axis='y', labelcolor=color_accuracy)
        
        # Set y-axis ranges
        overhead_max = max(overhead_mean) + max(overhead_std)
        ax1.set_ylim(0.8, overhead_max * 1.1)
        ax2.set_ylim(90, 105)
        
        # Set title
        model_name = MODEL_NAMES.get(model_key, model_key)
        # total_layers = config_data["total_layers"]
        ax1.set_title(f"{model_name}", fontweight='bold', fontsize=fontsize+2)
        
        # Apply aesthetics to primary axis only
        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='major', direction='in', 
                       length=8, width=2, top=True, right=False)
        ax1.tick_params(axis='both', which='minor', direction='in', 
                       length=4, width=1.5, top=True, right=False)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
        
        # Collect handles for legend (only from first subplot)
        if idx == 0:
            all_overhead_handles = [h1]
            all_accuracy_handles = h2
    
    # Create combined legend
    all_handles = all_overhead_handles + all_accuracy_handles
    all_labels = [f'{TOOL} Provenance Time', f'{TOOL} Attribution Accuracy']
    
    fig.legend(all_handles, all_labels, loc='upper center',
              bbox_to_anchor=(0.5, 1), ncol=2, frameon=True,
              prop={'weight': 'bold', 'size': fontsize}, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, "overhead-and-tool-accuracy")


if __name__ == "__main__":
    print("="*80)
    print("RQ3: Plotting Overhead vs Accuracy")
    print("="*80)
    
    print("\nLoading data...")
    data = load_overhead_data()
    print(f"Loaded data for {len(data)} model configurations")
    
    # Print comprehensive statistics for verification
    print_overhead_statistics(data)
    
    print("\nGenerating dual y-axis plots...")
    plot_dual_axis_overhead(data)
    
    print("\n✓ Plot generated successfully!")
    print(f"Output saved to: {OUTPUT_DIR.absolute()}")
