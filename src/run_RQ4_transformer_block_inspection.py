from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.model import get_model_and_tokenizer
from src.dataset.datasets import get_datasets_dict
from src.utils.generate import find_inputs_ids_where_response_is_correct, prepare_prompt
from src.provenance.fl_prov import ProvTextGenerator
from pathlib import Path
import logging
import argparse
import copy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RQ4: Middle Block Component Analysis')
parser.add_argument('--log', default='INFO', choices=['DEBUG', 'INFO'])
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--round_num', type=int, default=10)

args = parser.parse_args()

logger = logging.getLogger("RQ4")
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


def generate_middle_block_component_configs(model):
    total_layers = get_total_model_layers(model)
    middle_block = total_layers // 2
    
    layer_prefix = 'model'
    for name, _ in model.named_modules():
        if '.layers.' in name and 'self_attn' in name:
            layer_prefix = name.split('.layers.')[0]
            break
    
    logger.info(f"Total layers: {total_layers}, Middle block: {middle_block}, Prefix: {layer_prefix}")
    
    configs = {}
    
    attention_components = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    for comp in attention_components:
        layer_name = f'{layer_prefix}.layers.{middle_block}.self_attn.{comp}'
        configs[f'attn_{comp}'] = {
            'name': f'attn_{comp}',
            'prov_layers_names': [layer_name]
        }
    
    configs['mlp'] = {
        'name': 'mlp',
        'prov_layers_names': [f'{layer_prefix}.layers.{middle_block}.mlp']
    }
    
    configs['combined_protoken'] = {
        'name': 'combined_protoken',
        'prov_layers_names': [
            f'{layer_prefix}.layers.{middle_block}.self_attn.o_proj',
            f'{layer_prefix}.layers.{middle_block}.mlp'
        ]
    }
    
    return configs, middle_block


def measure_component_provenance(global_model, client_models, tokenizer, test_samples, layer_config, max_new_tokens=32):
    correct = 0
    client_contributions = []
    
    for sample in test_samples:
        prompt = prepare_prompt(sample['messages'], tokenizer)
        result = ProvTextGenerator.generate_text(
            global_model, client_models, tokenizer, prompt, 
            layer_config, max_new_tokens=max_new_tokens
        )
        
        predicted = max(result['client2part'], key=result['client2part'].get)
        if predicted in [0, 1]:
            correct += 1
        
        client_contributions.append(result['client2part'])
    
    return (correct / len(test_samples)) * 100.0, client_contributions


def run_middle_block_analysis(exp_key, round_num, num_test_samples):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    _, tokenizer = get_model_and_tokenizer(train_config)

    ds_dict = get_datasets_dict(
        num_clients=train_config['fl']['num_clients'],
        **train_config['dataset']
    )
    
    global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(exp_key, round_num)
    logger.info(f"Loaded {len(client_models)} client models")

    component_configs, middle_block = generate_middle_block_component_configs(global_model)
    logger.info(f"Generated {len(component_configs)} component configs for block {middle_block}")

    poison_dataset = ds_dict['test']['poison']
    label2dataset = find_inputs_ids_where_response_is_correct(
        global_model, tokenizer, {'poison': poison_dataset}, 
        min_samples=num_test_samples + 2
    )
    
    test_samples = [
        {'messages': label2dataset['poison']['messages'][i]}
        for i in range(min(num_test_samples, len(label2dataset['poison']['messages'])))
    ]
    
    logger.info(f"Prepared {len(test_samples)} test samples")
    
    results = {}
    for config_name, layer_config in component_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing component: {config_name}")
        logger.info(f"Layers: {layer_config['prov_layers_names']}")
        logger.info(f"{'='*60}")
        
        accuracy, contributions = measure_component_provenance(
            copy.deepcopy(global_model), 
            copy.deepcopy(client_models),
            tokenizer,
            test_samples,
            layer_config
        )
        
        results[config_name] = {
            'accuracy': accuracy,
            'layer_names': layer_config['prov_layers_names'],
            'contributions': contributions
        }
        
        logger.info(f"Accuracy: {accuracy:.2f}%")
    
    return {
        'exp_key': exp_key,
        'round_num': round_num,
        'total_layers': get_total_model_layers(global_model),
        'middle_block': middle_block,
        'num_samples': len(test_samples),
        'results': results,
        'metadata': train_config
    }


def plot_component_comparison(results, save_path):
    component_names = []
    accuracies = []
    
    component_order = ['attn_q_proj', 'attn_k_proj', 'attn_v_proj', 'attn_o_proj', 'mlp', 'combined_protoken']
    
    for comp_name in component_order:
        if comp_name in results['results']:
            display_name = comp_name.replace('_', ' ').replace('attn ', 'Attn: ').replace('mlp', 'MLP')
            component_names.append(display_name)
            accuracies.append(results['results'][comp_name]['accuracy'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    colors = colors[:len(component_names)]
    
    bars = ax.bar(range(len(component_names)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, name in enumerate(component_names):
        if 'o proj' in name.lower() or 'MLP' in name or 'combined' in name.lower():
            bars[i].set_linewidth(3)
    
    ax.set_xticks(range(len(component_names)))
    ax.set_xticklabels(component_names, rotation=45, ha='right')
    ax.set_ylabel('Provenance Attribution Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Transformer Block Components', fontsize=12, fontweight='bold')
    
    model_name = results['exp_key'].split('][')[0].replace('[', '')
    ax.set_title(f'RQ4: Middle Block Component Analysis\n{model_name} (Block {results["middle_block"]})', 
                 fontsize=14, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {save_path}")


def run_single_experiment(exp_key, results_dir, round_num, num_test_samples):
    results = run_middle_block_analysis(exp_key, round_num, num_test_samples)

    save_path = results_dir / f"middle_block_analysis/{exp_key}_round{round_num}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, save_path)
    logger.info(f"Saved: {save_path}")

    plot_path = results_dir / f"middle_block_analysis/{exp_key}_components_round{round_num}.png"
    plot_component_comparison(results, plot_path)

    return results


if __name__ == "__main__":
    results_dir = Path("results/rq4-middle-block-analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("RQ4: Middle Block Component Analysis")
    logger.info("="*80)

    all_exp_keys = list(CacheManager.get_completed_experiments_keys())
    logger.info(f"\nFound {len(all_exp_keys)} completed experiments")

    selected = [
        key for key in all_exp_keys 
        if 'Backdoor-True' in key and 'coding' in key.lower() and any(
            m in key for m in ['google_gemma', 'HuggingFaceTB_SmolLM', 'meta-llama_Llama', 'Qwen_Qwen']
        )
    ]

    logger.info(f"Selected {len(selected)} coding dataset backdoor experiments")
    logger.info(f"\nParams: round={args.round_num}, samples={args.num_samples}")

    _ = input("\nPress Enter to start...")

    all_results = {}
    for i, exp_key in enumerate(selected):
        logger.info(f"\n{'#'*80}")
        logger.info(f"[{i+1}/{len(selected)}] {exp_key}")
        logger.info(f"{'#'*80}")

        results = run_single_experiment(exp_key, results_dir, args.round_num, args.num_samples)
        all_results[exp_key] = results

    summary = results_dir / f"middle_block_analysis/summary_round{args.round_num}.json"
    save_json(all_results, summary)
    logger.info(f"\nSaved summary: {summary}")
    logger.info("\n" + "="*80)
    logger.info("RQ4 Complete!")
    logger.info("="*80)
