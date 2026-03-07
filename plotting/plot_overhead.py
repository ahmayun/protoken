import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plotting.common import (
    TOOL,
    MODEL_NAMES,
    setup_plot_style,
    save_figure,
)

def _parse_args():
    p = argparse.ArgumentParser(description="Plot RQ3 overhead (provenance time vs layers).")
    p.add_argument("--round_num", type=int, default=10, help="Round number for summary file")
    p.add_argument("--results_dir", required=True, help="Dir containing overhead_analysis/summary_round{N}.json")
    p.add_argument("--output_dir", required=True, help="Directory for output figure.")
    return p.parse_args()


def load_overhead_data(data_file):
    with open(data_file, "r") as f:
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


def _get_single_config_data(data):
    """Summary JSON has one key per experiment; return the single config (one model-dataset)."""
    keys = list(data.keys())
    if not keys:
        raise ValueError("Overhead summary JSON is empty")
    if len(keys) > 1:
        print(f"Note: summary has {len(keys)} experiments, using first: {keys[0]}")
    return keys[0], data[keys[0]]


def print_overhead_statistics(config_key, config_data):
    """Print overhead statistics for a single model-dataset config."""
    print("\n" + "=" * 80)
    print("RQ3 OVERHEAD STATISTICS")
    print("=" * 80)

    model_name = config_key.split("][")[0].replace("[", "") if "][" in config_key else config_key
    model_name = MODEL_NAMES.get(model_name, model_name)
    total_layers = config_data["total_layers"]
    results = config_data["results"]

    sorted_keys = sorted(results.keys(), key=lambda x: results[x]["num_layers"])
    min_config = results[sorted_keys[0]]
    max_config = results[sorted_keys[-1]]

    min_layers = min_config["num_layers"]
    max_layers = max_config["num_layers"]
    min_overhead = min_config["prov_mean"]
    max_overhead = max_config["prov_mean"]

    overhead_increase_pct = ((max_overhead - min_overhead) / min_overhead) * 100
    layer_increase_pct = ((max_layers - min_layers) / min_layers) * 100

    all_accuracy = [results[k]["accuracy"] for k in results.keys()]
    accuracy_check = "✓" if all(acc == 100.0 for acc in all_accuracy) else "✗"

    print(f"\n{model_name} ({total_layers} total layers):")
    print(f"  Overhead Range:")
    print(f"    Min: {min_overhead:.2f}s with {min_layers} layers")
    print(f"    Max: {max_overhead:.2f}s with {max_layers} layers")
    print(f"  Overhead Increase: {overhead_increase_pct:.0f}% for {layer_increase_pct:.0f}% layer increase")
    print(f"  Accuracy: 100% across all {len(results)} configurations {accuracy_check}")
    print(f"  Detailed configurations:")
    for config_name in sorted_keys:
        config = results[config_name]
        print(f"    {config['num_layers']:2d} layers: {config['prov_mean']:.2f}s (±{config['prov_std']:.3f}), Acc: {config['accuracy']:.0f}%")
    print("\n" + "=" * 80 + "\n")


def plot_dual_axis_overhead(config_key, config_data, output_dir):
    """Create a single dual y-axis plot for overhead vs accuracy (one model-dataset)."""
    fontsize = 18
    setup_plot_style(font_size=fontsize)

    num_layers, overhead_mean, overhead_std, accuracy = extract_model_data(config_data)

    model_name = config_key.split("][")[0].replace("[", "") if "][" in config_key else config_key
    model_name = MODEL_NAMES.get(model_name, model_name)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    color_overhead = "Blue"
    h1 = ax1.errorbar(
        num_layers, overhead_mean, yerr=overhead_std,
        marker="o", linewidth=2.5, markersize=8,
        color=color_overhead, capsize=5, capthick=2,
        label=f"{TOOL} Provenance Time", zorder=3,
    )
    ax1.set_xlabel("Number of Layers", fontweight="bold")
    ax1.set_ylabel("Avg. Provenance Time (s)", fontweight="bold", color=color_overhead)
    ax1.tick_params(axis="y", labelcolor=color_overhead)

    ax2 = ax1.twinx()
    color_accuracy = "k"
    h2 = ax2.plot(
        num_layers, accuracy, marker="s", linewidth=2.5, markersize=8,
        color=color_accuracy, label=f"{TOOL} Accuracy", linestyle="--", zorder=3,
    )
    ax2.set_ylabel(f"{TOOL} Accuracy (%)", fontweight="bold", color=color_accuracy)
    ax2.tick_params(axis="y", labelcolor=color_accuracy)

    overhead_max = max(overhead_mean) + max(overhead_std)
    ax1.set_ylim(0.8, overhead_max * 1.1)
    ax2.set_ylim(90, 105)

    ax1.set_title(f"RQ3: Provenance Time vs. Accuracy\n{model_name}", fontweight="bold", fontsize=fontsize + 2)
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", direction="in", length=8, width=2, top=True, right=False)
    ax1.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5, top=True, right=False)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)

    fig.legend([h1, h2[0]], [f"{TOOL} Provenance Time", f"{TOOL} Attribution Accuracy"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True,
               prop={"weight": "bold", "size": fontsize}, framealpha=0.9)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "overhead-and-tool-accuracy", output_dir=output_dir)


if __name__ == "__main__":
    args = _parse_args()
    results_dir = pathlib.Path(args.results_dir)
    data_file = results_dir / "overhead_analysis" / f"summary_round{args.round_num}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Overhead summary not found: {data_file}")

    print("=" * 80)
    print("RQ3: Plotting Overhead vs Accuracy")
    print("=" * 80)
    print(f"\nLoading: {data_file}")
    data = load_overhead_data(data_file)
    config_key, config_data = _get_single_config_data(data)

    print_overhead_statistics(config_key, config_data)

    print("Generating dual y-axis plot...")
    plot_dual_axis_overhead(config_key, config_data, args.output_dir)

    print("\n✓ Plot generated successfully!")
    print(f"Output saved to: {pathlib.Path(args.output_dir).absolute()}")
