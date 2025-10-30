from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.model import get_model_and_tokenizer
from src.dataset.datasets import get_datasets_dict
from src.utils.generate import find_inputs_ids_where_response_is_correct, prepare_prompt, generate_text
from src.provenance.fl_prov import ProvTextGenerator
from pathlib import Path
import logging
import argparse
import copy
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RQ3: Computational Overhead Analysis')
parser.add_argument('--log', default='INFO', choices=['DEBUG', 'INFO'])
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--round_num', type=int, default=10)
parser.add_argument('--layer_interval', type=int, default=2)

args = parser.parse_args()

logger = logging.getLogger("RQ3")
logger.setLevel(args.log.upper())
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_total_model_layers(model):
    max_layer_idx = -1
    for name, _ in model.named_modules():
        for i in range(1000):
            if name.endswith(f'.layers.{i}'):
                max_layer_idx = max(max_layer_idx, i)
    return max_layer_idx + 1


def generate_layer_configs(total_layers, interval=2):
    configs = {}
    for num_layers in range(interval+1, total_layers + 1, interval):
        configs[f'last_{num_layers}'] = {
            'name': f'last_{num_layers}',
            'patterns': ['self_attn.o_proj', '.mlp', 'lm_head'],
            'exclude_patterns': [],
            'last_n': num_layers
        }
    
    if total_layers not in range(interval, total_layers + 1, interval):
        configs[f'last_{total_layers}'] = {
            'name': f'last_{total_layers}',
            'patterns': ['self_attn.o_proj', '.mlp', 'lm_head'],
            'exclude_patterns': [],
            'last_n': total_layers
        }
    
    return configs


def measure_baseline_time(model, tokenizer, test_samples, max_new_tokens=32):
    model.eval()
    times = []
    
    for sample in test_samples:
        prompt = prepare_prompt(sample['messages'], tokenizer)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        _ = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times)


def measure_provenance_time(global_model, client_models, tokenizer, test_samples, layer_config, max_new_tokens=32):
    times = []
    correct = 0
    
    for sample in test_samples:
        prompt = prepare_prompt(sample['messages'], tokenizer)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        result = ProvTextGenerator.generate_text(
            global_model, client_models, tokenizer, prompt, 
            layer_config, max_new_tokens=max_new_tokens
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
        
        predicted = max(result['client2part'], key=result['client2part'].get)
        if predicted in [0, 1]:
            correct += 1
    
    return np.mean(times), np.std(times), (correct / len(test_samples)) * 100.0


def run_overhead_analysis(exp_key, layer_configs, round_num, num_test_samples):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    _, tokenizer = get_model_and_tokenizer(train_config)

    ds_dict = get_datasets_dict(
        num_clients=train_config['fl']['num_clients'],
        **train_config['dataset']
    )
    
    global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(exp_key, round_num)
    logger.info(f"Loaded {len(client_models)} client models")

    poison_dataset = ds_dict['test']['poison']
    label2dataset = find_inputs_ids_where_response_is_correct(
        global_model, tokenizer, {'poison': poison_dataset}, 
        min_samples=num_test_samples + 2
    )
    
    test_samples = []
    for i in range(min(num_test_samples, len(label2dataset['poison']['messages']))):
        test_samples.append({'messages': label2dataset['poison']['messages'][i]})
    
    logger.info(f"Prepared {len(test_samples)} test samples")
    
    logger.info("\nMeasuring baseline...")
    baseline_mean, baseline_std = measure_baseline_time(global_model, tokenizer, test_samples)
    logger.info(f"Baseline: {baseline_mean:.4f}s ± {baseline_std:.4f}s")
    
    results = {}
    for config_name, layer_config in layer_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"Tracking last {layer_config['last_n']} transformer blocks")
        logger.info(f"{'='*60}")
        
        prov_mean, prov_std, accuracy = measure_provenance_time(
            copy.deepcopy(global_model), 
            copy.deepcopy(client_models),
            tokenizer,
            test_samples,
            layer_config
        )
        
        overhead = prov_mean - baseline_mean
        overhead_pct = (overhead / baseline_mean) * 100.0
        
        results[config_name] = {
            'num_layers': layer_config['last_n'],
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'prov_mean': prov_mean,
            'prov_std': prov_std,
            'overhead': overhead,
            'overhead_pct': overhead_pct,
            'accuracy': accuracy
        }
        
        logger.info(f"Provenance: {prov_mean:.4f}s ± {prov_std:.4f}s")
        logger.info(f"Overhead: {overhead_pct:.2f}% ({overhead:.4f}s)")
        logger.info(f"Accuracy: {accuracy:.2f}%")
    
    return {
        'exp_key': exp_key,
        'round_num': round_num,
        'total_layers': get_total_model_layers(global_model),
        'num_samples': len(test_samples),
        'results': results,
        'metadata': train_config
    }


def plot_overhead_accuracy(results, save_path):
    sorted_results = sorted(results['results'].items(), key=lambda x: x[1]['num_layers'])
    
    num_layers = [r[1]['num_layers'] for r in sorted_results]
    overhead = [r[1]['overhead_pct'] for r in sorted_results]
    accuracy = [r[1]['accuracy'] for r in sorted_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Number of Layers Tracked', fontsize=12)
    ax1.set_ylabel('Computational Overhead (%)', color=color1, fontsize=12)
    line1 = ax1.plot(num_layers, overhead, marker='o', color=color1, linewidth=2, markersize=8, label='Overhead')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Provenance Accuracy (%)', color=color2, fontsize=12)
    line2 = ax2.plot(num_layers, accuracy, marker='s', color=color2, linewidth=2, markersize=8, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    model_name = results['exp_key'].split('][')[0].replace('[', '')
    plt.title(f'RQ3: Overhead vs. Accuracy Tradeoff\n{model_name}', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.text(0.98, 0.02, f'Total layers: {results["total_layers"]}', 
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {save_path}")


def run_single_experiment(exp_key, results_dir, round_num=10, num_test_samples=5, layer_interval=3):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    temp_model, _ = get_model_and_tokenizer(train_config)

    total_layers = get_total_model_layers(temp_model)
    logger.info(f"Detected {total_layers} transformer layers")

    layer_configs = generate_layer_configs(total_layers, layer_interval)
    logger.info(f"Generated {len(layer_configs)} layer configs at {layer_interval}-layer intervals")

    results = run_overhead_analysis(exp_key, layer_configs, round_num, num_test_samples)

    save_path = results_dir / f"overhead_analysis/{exp_key}_round{round_num}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, save_path)
    logger.info(f"Saved: {save_path}")

    plot_path = results_dir / f"overhead_analysis/{exp_key}_overhead_round{round_num}.png"
    plot_overhead_accuracy(results, plot_path)

    return results


if __name__ == "__main__":
    results_dir = Path("results/rq3-overhead")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("RQ3: Computational Overhead Analysis")
    logger.info("="*80)

    all_exp_keys = list(CacheManager.get_completed_experiments_keys())
    logger.info(f"\nFound {len(all_exp_keys)} completed experiments")

    selected = [
        key for key in all_exp_keys 
        if 'Backdoor-True' in key and any(
            m in key for m in ['google_gemma', 'HuggingFaceTB_SmolLM', 'meta-llama_Llama', 'Qwen_Qwen']
        )
    ]

    logger.info(f"Selected {len(selected)} backdoor experiments")
    logger.info(f"\nParams: round={args.round_num}, samples={args.num_samples}, interval={args.layer_interval}")

    _ = input("\nPress Enter to start...")

    all_results = {}
    for i, exp_key in enumerate(selected):
        logger.info(f"\n{'#'*80}")
        logger.info(f"[{i+1}/{len(selected)}] {exp_key}")
        logger.info(f"{'#'*80}")

        results = run_single_experiment(
            exp_key, results_dir, args.round_num, args.num_samples, args.layer_interval
        )
        all_results[exp_key] = results

    summary = results_dir / f"overhead_analysis/summary_round{args.round_num}.json"
    save_json(all_results, summary)
    logger.info(f"\nSaved summary: {summary}")
    logger.info("\n" + "="*80)
    logger.info("RQ3 Complete!")
    logger.info("="*80)
