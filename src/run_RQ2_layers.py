from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.model import get_model_and_tokenizer
from src.dataset.datasets import get_datasets_dict
from src.utils.generate import find_inputs_ids_where_response_is_correct
from src.run_provenance import FL_Provenance
from src.provenance.fl_prov import get_all_layers
from plotting.plot_rq2_individual_layers import plot_individual_layer_accuracy
from pathlib import Path
import logging
import argparse
import copy

parser = argparse.ArgumentParser(description='RQ2: Individual Layer Testing')
parser.add_argument('--log', default='INFO',
                    choices=['DEBUG', 'INFO'], help='Logging level')
parser.add_argument('--num_samples', type=int, default=5,
                    help='Number of test samples')
parser.add_argument('--round_num', type=int, default=10,
                    help='Which FL round to evaluate')

args = parser.parse_args()

logger = logging.getLogger("RQ2")
logger.setLevel(args.log.upper())
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_total_model_layers(model):
    max_layer_idx = -1
    all_layers_names = []
    for name, _ in model.named_modules():
        all_layers_names.append(name)

    for i in range(1000000000):
        block = f'.layers.{i}'
        if any(lname.endswith(block) for lname in all_layers_names):
            max_layer_idx = i
        else:
            break

    return max_layer_idx + 1


def _get_layer_patterns(config_key, layer_idx):
#      model.layers.23.self_attn.q_proj        Linear              False        803712            803712
# 209          model.layers.23.self_attn.k_proj        Linear              False        114816            114816
# 210          model.layers.23.self_attn.v_proj        Linear              False        114816            114816
# 211          model.layers.23.self_attn.o_proj        Linear              False        802816            802816
# 212             model.layers.23.mlp.gate_proj        Linear              False       4358144           4358144
# 213               model.layers.23.mlp.up_proj        Linear              False       4358144           4358144
# 214             model.layers.23.mlp.down_proj

    a = {
        'name': config_key,
        'prov_layers_names': [
            # f'model.layers.{layer_idx}.self_attn.q_proj',
            # f'model.layers.{layer_idx}.self_attn.k_proj',
            # f'model.layers.{layer_idx}.self_attn.v_proj',
            f'model.layers.{layer_idx}.self_attn.o_proj',
            # f'model.layers.{layer_idx}.mlp.gate_proj',
            # f'model.layers.{layer_idx}.mlp.up_proj',
            f'model.layers.{layer_idx}.mlp',
            # 'lm_head'
        ],
        'patterns': [],
        'exclude_patterns': [],
        'last_n': None
    }

    
    return a 


def generate_individual_layer_configs(total_layers):

    configs = {}
    layer_idx = 0
    for layer_idx in range(total_layers):
        config_key = f'layer_{layer_idx}'
        configs[config_key] = _get_layer_patterns(config_key, layer_idx)
    
    layer_idx = len(configs)

    configs[f'layer_{layer_idx}'] = {
        'name': 'lm_head_only',
        'prov_layers_names': ['lm_head'],
        'patterns': [],
        'exclude_patterns': [],
        'last_n': None}
    return configs


def run_individual_layer_sweep(exp_key, layer_configs, round_num, num_test_samples):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    _, tokenizer = get_model_and_tokenizer(train_config)

    ds_dict = get_datasets_dict(
        num_clients=train_config['fl']['num_clients'],
        **train_config['dataset']
    )
    test_dataset_dict = ds_dict['test']
    client_labels = ds_dict['client_labels']

    global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(
        exp_key, round_num)
    
    

    logger.info(
        f"Loaded {len(client_models)} client models. Client2Labels: {client_labels}")

    clients2labels = {cid: client_labels[cid] for cid in client_models.keys()}
    unique_labels_across_clients = set()
    for labels in clients2labels.values():
        unique_labels_across_clients.update(labels)

    unique_labels_across_clients = list(unique_labels_across_clients)
    logger.info(f'Participating Clients labels: {clients2labels}')
    logger.info(
        f'Unique labels across participating clients: {unique_labels_across_clients}')

    if 'poison' in unique_labels_across_clients:
        logger.info(
            'Evaluating Backdoored FL Setting. Malicious client present')
    else:
        raise ValueError("'poison' should be in unique labels")

    label2dataset = find_inputs_ids_where_response_is_correct(
        global_model, tokenizer, {'poison': test_dataset_dict['poison']}, min_samples=num_test_samples+2)

    results_per_config = {}

    for config_name, layer_config in layer_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"Layers: {layer_config['prov_layers_names']}")
        logger.info(f"{'='*60}")

        with FL_Provenance(copy.deepcopy(global_model), copy.deepcopy(client_models), tokenizer, layer_config) as fl_prov:
            config_results = fl_prov.run_provenance_on_samples(
                label2dataset, num_samples=num_test_samples, client_labels=clients2labels
            )

        if config_results is None:
            logger.warning(f"No results for {config_name}, skipping...")
            continue

        layer_idx = config_name.replace('layer_', '').replace('_only', '')
        results_per_config[config_name] = {
            'layer_index': int(layer_idx),
            'overall_accuracy': config_results['overall_accuracy'],
            'per_client_accuracy': config_results['per_client_accuracy'],
            'detailed_results': config_results['detailed_results']
        }

        logger.info(
            f"{config_name}: Accuracy = {config_results['overall_accuracy']:.2f}%")

    total_num_layers = get_total_model_layers(global_model)
    return {
        'experiment_key': exp_key,
        'round_num': round_num,
        'total_model_layers': total_num_layers,
        'layer_configs_results': results_per_config,
        'metadata': {
            'training_config': train_config
        }
    }


def run_single_experiment(exp_key, results_dir, round_num=10, num_test_samples=5):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    temp_model, _  = get_model_and_tokenizer(train_config)
    
    total_layers = get_total_model_layers(temp_model)
    logger.info(f"Detected {total_layers} transformer layers for this model")
    
    layer_configs = generate_individual_layer_configs(total_layers)

    # for cfg_key, cfg in layer_configs.items():
    #     logger.info(f"Layer Config: {cfg_key} -> {cfg['prov_layers_names']}")
    #     temp_layers = get_all_layers(temp_model, cfg)
    #     for l in temp_layers:
    #         logger.info(f"  - {l}") 


        


    # _ = input("Press Enter to start individual layer sweep...")
    
    results = run_individual_layer_sweep(
        exp_key=exp_key,
        layer_configs=layer_configs,
        round_num=round_num,
        num_test_samples=num_test_samples
    )
    
    save_path = results_dir/f"individual_layers/{exp_key}_round{round_num}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, save_path)
    logger.info(f"Saved: {save_path}")
    
    plot_individual_layer_accuracy(save_path)
    logger.info(f"Generated plot for: {exp_key}")
    
    return results




if __name__ == "__main__":
    
    results_dir = Path("results/rq2-5")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("RQ2 Experiment: Individual Layer Testing")
    logger.info("="*80)

    logger.info("\nAvailable completed experiments:")
    all_exp_keys = list(CacheManager.get_completed_experiments_keys())
    for i, key in enumerate(all_exp_keys):
        logger.info(f"[{i}] {key}")
    

    # selected_exp_keys = [key for key in all_exp_keys if 'Backdoor-True' in key]
    # logger.info(f"\nRunning on ALL {len(selected_exp_keys)} backdoor experiments")
    
    # key = "[Qwen_Qwen2.5-0.5B-Instruct][rounds-10][epochs-1][clients6-per-round-6][Datasets-['math']-None][partitioning-iid][Backdoor-True][Unsloth-False][Lora-False]"
    key = "[google_gemma-3-270m-it][rounds-10][epochs-1][clients6-per-round-6][Datasets-['coding']-None][partitioning-iid][Backdoor-True][Unsloth-False][Lora-False]"
    selected_exp_keys = [key]
    logger.info(f"\nRunning on single experiment (use --all for all experiments)")

    logger.info(f"\nParameters:")
    logger.info(f"  - Results directory: {results_dir}")
    logger.info(f"  - Round number: {args.round_num}")
    logger.info(f"  - Test samples per layer: {args.num_samples}")
    logger.info(f"  - Total experiments: {len(selected_exp_keys)}")

    _ = input("\nPress Enter to start individual layer testing...")

    all_results = {}
    
    for i, exp_key in enumerate(selected_exp_keys):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"[{i+1}/{len(selected_exp_keys)}] Processing: {exp_key}")
        logger.info(f"{'#'*80}")
        
        results = run_single_experiment(
            exp_key=exp_key,
            results_dir=results_dir,
            round_num=args.round_num,
            num_test_samples=args.num_samples
        )
        
        all_results[exp_key] = results
    
    summary_path = results_dir/f"individual_layers/all_experiments_summary_round{args.round_num}.json"
    save_json(all_results, summary_path)
    logger.info(f"\nSaved summary: {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("RQ2 Individual Layer Testing Complete!")
    logger.info("="*80)
