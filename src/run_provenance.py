from src.utils.judge import llm_judge
from src.provenance.fl_prov import ProvTextGenerator
from src.utils.plotting import plot_federated_metrics
from src.utils.cache import CacheManager
from src.utils.utils import save_json
from src.fl.model import get_model_and_tokenizer
from src.dataset.datasets import get_datasets_dict
from pathlib import Path
import torch
import gc
import logging
from typing import Dict, List
import argparse

import logging
import argparse

# --- Argument Parsing (No changes needed here) ---
parser = argparse.ArgumentParser(
    description='A simple script with adjustable logging.')
parser.add_argument(
    '--log',
    default='INFO',
    choices=['DEBUG', 'INFO'],
    help='Set the logging level for the Prov logger (default: INFO)'
)
args = parser.parse_args()


# 1. Get your specific logger by name
logger = logging.getLogger("Prov")

# 2. Set its level from the command-line argument
logger.setLevel(args.log.upper())

# 3. Create a handler to send logs to the console
handler = logging.StreamHandler()

# 4. (Optional) Create a format and add it to the handler
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)

# 5. Add the configured handler to your logger
logger.addHandler(handler)


MODEL2LayerConfig = {
    'lora': {
        'name': 'lora',
        # 'patterns': ['.self_attn.o_proj.lora_A.default', 'self_attn.o_proj.lora_B.default'
        #              '.mlp.gate_proj.lora_A.default', '.mlp.gate_proj.lora_B.default',
        #              '.mlp.up_proj.lora_A.default', '.mlp.up_proj.lora_B.default',
        #              'mlp.down_proj.lora_A.default', 'mlp.down_proj.lora_B.default'
        #              ],
        'patterns': ['.lora_A.default', '.lora_B.default'],
        # 'patterns': ['.mlp'],
        'exclude_patterns': ['lora_dropout'],
        'last_n': 2
    },
    'standard': {
        'name': 'mlp',
        'patterns': ['.mlp.gate_proj', '.mlp.up_proj', '.mlp.down_proj'],
        'exclude_patterns': [],
        'last_n': 3
    }
}


class FL_Provenance:
    def __init__(self, global_model, client_models, tokenizer, layer_config):
        self.global_model = global_model
        self.client_models = client_models
        self.tokenizer = tokenizer
        self.layer_config = layer_config

    def run_provenance_on_samples(self, dataset_dict, num_samples, client_labels):
        per_client_accuracy = {}
        detailed_results = []

        for ds_name, dataset in dataset_dict.items():
            dataset = dataset.shuffle()
            correct_predictions = 0
            sample_indices = list(
                range(min(num_samples, len(dataset['messages']))))

            for sample_idx in sample_indices:
                logger.info(
                    f'{10*'='} Provenance of Input Id {sample_idx}  {'='*10}\n')

                sample_result = self._analyze_single_sample(
                    dataset['messages'][sample_idx], ds_name, client_labels)
                detailed_results.append(sample_result)

                if sample_result['is_correct']:
                    correct_predictions += 1

                logger.info(f"{'*'*30}\n")

            accuracy = (correct_predictions / len(sample_indices)) * 100.0
            per_client_accuracy[ds_name] = accuracy

        overall_accuracy = sum(per_client_accuracy.values()
                               ) / len(per_client_accuracy)

        logger.info(f'Avg. Accuracy = {overall_accuracy}, '
                    f'Per Client Accuracy = {per_client_accuracy}'
                    )

        return {
            'per_client_accuracy': per_client_accuracy,
            'overall_accuracy': overall_accuracy,
            'detailed_results': detailed_results
        }

    def _analyze_single_sample(self, conversation, actual_label, client_labels):
        prompt = self._prepare_prompt(conversation)
        
        result = ProvTextGenerator.generate_text(
            self.global_model, self.client_models, self.tokenizer, prompt, self.layer_config, max_new_tokens=32)

        # print("\nFull Conversation:")
        # print(conversation)
        logger.debug("\n\n")
        logger.debug(f">>Input Prompt is following:\n{prompt.replace('\n', '').strip()}")
        logger.debug("\n\n")
        logger.debug(f">> Generated Response is following:\n{result['response']}")
        logger.debug("\n\n")
        logger.debug(f">> Actual Response is following:\n{conversation[-1]['content']}\n\n")

        assert len(result['client2part']
                   ) > 1, f"Total clients are {len(result['client2part'])}"

        predicted_client = max(
            result['client2part'], key=result['client2part'].get)

        # Check if predicted client is responsible for this label
        predicted_client_labels = client_labels[predicted_client]
        assert isinstance(predicted_client_labels,
                          list), f"Predicted client labels should be a list, got {type(predicted_client_labels)}"

        is_correct = actual_label in predicted_client_labels

        # Organized logging with f-strings
        logger.info(
            f"Layer Config: {self.layer_config['name']}\n"
            f"Actual Label: {actual_label}\n"
            f"Predicted Client-{predicted_client} has labels {predicted_client_labels}\n"
            f"Is Correct: {is_correct}\n"
            f"{result['client2part']}\n"
            f"clients2labels: {client_labels}\n"
        )

        return {
            'predicted_client': predicted_client,
            'client2part': result['client2part'],
            'is_correct': is_correct,
            'actual_label': actual_label,
            'predicted_client_labels': predicted_client_labels,
        }

    def _prepare_prompt(self, conversation: List[Dict]) -> str:
        messages = [
            {'role': conversation[0]['role'],
                'content': conversation[0]['content']},
            {'role': conversation[1]['role'],
                'content': conversation[1]['content']}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).removeprefix('<bos>')

        return text

    def cleanup(self):
        if hasattr(self, 'client_models') and self.client_models:
            for model in self.client_models.values():
                if model is not None:
                    model.cpu()
            self.client_models.clear()
            del self.client_models

        if hasattr(self, 'global_model') and self.global_model is not None:
            self.global_model.cpu()
            del self.global_model

        torch.cuda.empty_cache()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def single_round_provenance_refactored(exp_key: str, round_num: int,
                                       dataset_dict: dict, client_labels: dict, tokenizer, layer_config, num_test_samples):
    logger.info(f"{10*'-'} Round {round_num} {10*'-'}\n\n")

    global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(
        exp_key, round_num)

    clients2labels = {cid: client_labels[cid] for cid in client_models.keys()}
    unique_labels_across_clients = set()
    for labels in clients2labels.values():
        unique_labels_across_clients.update(labels)

    unique_labels_across_clients = list(unique_labels_across_clients)
    logger.info(f'Participating Clients labels: {clients2labels}')
    logger.info(f'Dataset Dict keys {list(dataset_dict.keys())}')
    logger.info(
        f'Unique labels across participating clients: {unique_labels_across_clients}')
    label2dataset = {k: dataset_dict[k] for k in unique_labels_across_clients}
    _ = input("Press Enter to continue...")

    assert len(unique_labels_across_clients) == len(
        label2dataset), f"Unique labels {unique_labels_across_clients}, dataset dict keys {list(dataset_dict.keys())}"

    assert len(
        unique_labels_across_clients) > 1, f"Unique labels {unique_labels_across_clients}, dataset dict keys {list(dataset_dict.keys())}"

    with FL_Provenance(global_model, client_models, tokenizer, layer_config) as fl_prov:
        results = fl_prov.run_provenance_on_samples(
            label2dataset, num_samples=num_test_samples, client_labels=clients2labels)

    return results


def rounds_provenance_refactored(exp_key, num_test_samples):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    temp_model, tokenizer = get_model_and_tokenizer(train_config)

    if hasattr(temp_model, 'peft_config'):
        layers_config = MODEL2LayerConfig['lora']
    else:
        layers_config = MODEL2LayerConfig['standard']

    logger.info(f'Layers Config: {layers_config}')

    datasets_result = get_datasets_dict(
        num_clients=train_config['fl']['num_clients'],  **train_config['dataset'])
    dataset_dict = datasets_result['test']
    client_labels = datasets_result['client_labels']

    round2provenance = {}
    total_rounds = train_config['fl']['num_rounds']

    across_all_rounds_accuracy = 0.0
    count = 0
    for round_num in range(1, total_rounds+1):
        # for round_num in [15]:
        summary_stats = single_round_provenance_refactored(
            exp_key, round_num, dataset_dict, client_labels, tokenizer, layers_config, num_test_samples)
        round2provenance[round_num] = summary_stats
        across_all_rounds_accuracy += summary_stats['overall_accuracy']
        count += 1

    across_all_rounds_accuracy = across_all_rounds_accuracy/count

    logger.info(
        f'\n\n {"#"*10} Accross All Rounds Accuracy = {across_all_rounds_accuracy} {"#"*10}')

    return {
        'across_all_rounds_accuracy': across_all_rounds_accuracy,
        "provenance": round2provenance,
        'metadata': {'training_config': train_config}
    }


def full_cache_provenance(results_dir: Path, num_test_samples: int = 5):
    print(f"{10*'-'} Running Refactored Provenance Analysis {10*'-'}")

    for i, exp_key in enumerate(CacheManager.get_completed_experiments_keys()):
        print(f"[{i}] Key: {exp_key}")

    _ = input("\nPress Enter to start processing all experiment keys...")

    for i, exp_key in enumerate(CacheManager.get_completed_experiments_keys()):
        json_path = results_dir / f"provenance_refactored_{exp_key}.json"
        if json_path.exists():
            print(f"\nProvenance info already exists: {json_path}")
            print(f'\n\n {10*"="}')
            continue

        print(
            f"\n\n [{i}] Running provenance analysis for experiment key: {exp_key}")

        try:
            result = rounds_provenance_refactored(
                exp_key=exp_key, num_test_samples=num_test_samples)
        except Exception as e:
            logger.error(f"Error processing {exp_key}: {e}")
            continue

        CacheManager.set_provenance_results(exp_key, result)
        result['training'] = CacheManager.load_training_metrics(
            exp_key=exp_key)

        save_json(result, json_path)
        plot_federated_metrics(json_file_path=json_path,
                               save_fig_path=results_dir/f"plot_{exp_key}.png")


def single_key_provenance(results_dir: Path, num_test_samples: int = 5):
 
    exp_key = "[google_gemma-3-1b-pt][rounds-5][epochs-1][clients25-per-round-3][['medical', 'finance', 'math']-1][Lora-False]"
    json_path = results_dir / f"single_provenance_refactored_{exp_key}.json"

    prov_dict = rounds_provenance_refactored(
        exp_key=exp_key, num_test_samples=num_test_samples)
    CacheManager.set_provenance_results(exp_key, prov_dict)
    prov_dict['training'] = CacheManager.load_training_metrics(exp_key=exp_key)

    save_json(prov_dict, json_path)
    # plot_federated_metrics(json_file_path=json_path, save_fig_path=results_dir/f"plot_{exp_key}.png")
    plot_federated_metrics(json_file_path=json_path,
                           save_fig_path=results_dir/f"test1.png")


if __name__ == "__main__":

    results_dir = Path("results")
    # print(f"\n{10*'-'} Testing Different Layer Configs {10*'-'}")
    # single_key_provenance_refactored(results_dir)
    # while True:
    # full_cache_provenance(results_dir)
    #     time.sleep(10)
    single_key_provenance(Path("results_debug"))
