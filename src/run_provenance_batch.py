from src.utils.judge import llm_judge
from src.provenance.fl_prov import ProvTextGenerator
from src.utils.plotting import plot_federated_metrics
from src.utils.utils import save_json, CacheManager
from src.utils.model import get_model_and_tokenizer
from src.utils.datasets import get_datasets_dict
from pathlib import Path
import torch
import gc
import logging
from typing import Dict, List
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(f"Prov")


MODEL2LayerConfig = {
    'lora': {
        'name': 'lora',
        'patterns': ['.mlp.down_proj.lora_A.default'],
        'exclude_patterns': ['lora_dropout'],
        'last_n': 1
    },
    'standard': {
        'name': 'mlp',
        'patterns': ['.mlp'],
        'exclude_patterns': [],
        'last_n': 1
    }
}


class FL_Provenance:
    def __init__(self, global_model, client_models, tokenizer, layer_config):
        self.global_model = global_model
        self.client_models = client_models
        self.tokenizer = tokenizer
        self.layer_config = layer_config

    def run_provenance_on_samples(self, dataset_dict, num_samples):
        per_client_accuracy = {}
        detailed_results = []

        for expected_client_id, dataset in dataset_dict.items():
            correct_predictions = 0
            sample_indices = list(
                range(min(num_samples, len(dataset['messages']))))

            for sample_idx in sample_indices:
                conversation = dataset['messages'][sample_idx]
                sample_result = self._analyze_single_sample(
                    conversation, expected_client_id)
                detailed_results.append(sample_result)

                if sample_result['is_correct']:
                    correct_predictions += 1

            accuracy = (correct_predictions / len(sample_indices)) * 100.0
            per_client_accuracy[expected_client_id] = accuracy

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

    def _analyze_single_sample(self, conversation, expected_client_id):
        prompt = self._prepare_prompt(conversation)
        result = ProvTextGenerator.generate_text(
            self.global_model, self.client_models, self.tokenizer, prompt, self.layer_config)

        predicted_client = max(
            result['client2part'], key=result['client2part'].get)
        is_correct = predicted_client == expected_client_id

        # Organized logging with f-strings
        logger.info(
            f"Config: {self.layer_config['name']}, "
            f"Actual: {expected_client_id}, "
            f"Predicted: {predicted_client}, "
            f"Is Correct: {is_correct}, "
            f"Parts: {result['client2part']}"
        )

        # Debug logging for additional details
        logger.debug(f"Client contributions: {result['client2part']}")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        return {
            'predicted_client': predicted_client,
            'actual_client': expected_client_id,
            'client2part': result['client2part'],
            'is_correct': is_correct
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
                                       dataset_dict: dict, tokenizer, layer_config, num_test_samples: int = 10):
    logger.info(f"{10*'-'} Round {round_num} {10*'-'}")

    global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(
        exp_key, round_num)

    with FL_Provenance(global_model, client_models, tokenizer, layer_config) as fl_prov:
        results = fl_prov.run_provenance_on_samples(
            dataset_dict, num_samples=num_test_samples)

    return results


def rounds_provenance_refactored(exp_key, num_test_samples: int = 10):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    temp_model, tokenizer = get_model_and_tokenizer(train_config)

    if hasattr(temp_model, 'peft_config'):
        layers_config = MODEL2LayerConfig['lora']
    else:
        layers_config = MODEL2LayerConfig['standard']

    logger.info(f'Layers Config: {layers_config}')

    dataset_dict = get_datasets_dict(train_config['dataset'])['test']

    round2provenance = {}
    total_rounds = train_config['fl']['num_rounds']

    for round_num in range(1, total_rounds):
        summary_stats = single_round_provenance_refactored(
            exp_key, round_num, dataset_dict, tokenizer, layers_config, num_test_samples)
        round2provenance[round_num] = summary_stats

    return {
        "provenance": round2provenance,
        'metadata': {'training_config': train_config}
    }


def full_cache_provenance_refactored(results_dir: Path, num_test_samples: int = 10):
    print(f"{10*'-'} Running Refactored Provenance Analysis {10*'-'}")

    for exp_key in CacheManager.get_completed_experiments_keys():
        json_path = results_dir / f"provenance_refactored_{exp_key}.json"
        if json_path.exists():
            print(f"\nProvenance info already exists: {json_path}")
            print(f'\n\n {10*"="}')
            continue

        print(f"Running provenance analysis for experiment key: {exp_key}")
        result = rounds_provenance_refactored(
            exp_key=exp_key, num_test_samples=num_test_samples)

        CacheManager.set_provenance_results(exp_key, result)
        result['training'] = CacheManager.load_training_metrics(
            exp_key=exp_key)

        save_json(result, json_path)
        plot_provenance_accuracy(json_path, results_dir=results_dir)


def single_key_provenance_refactored(results_dir: Path, num_test_samples: int = 10):
    exp_key = "[google_gemma-3-270m-it][rounds16][epochs-2][clients2][C0-medical-C1finance][LoRA-r8-alpha8][New2]"
    json_path = results_dir / f"single_provenance_refactored_{exp_key}.json"

    # prov_dict = rounds_provenance_refactored(
    #     exp_key=exp_key, num_test_samples=num_test_samples)
    # CacheManager.set_provenance_results(exp_key, prov_dict)
    # prov_dict['training'] = CacheManager.load_training_metrics(exp_key=exp_key)

    # save_json(prov_dict, json_path)
    plot_federated_metrics(json_file_path=json_path, result_dir=results_dir)


if __name__ == "__main__":

    results_dir = Path("results")
    print(f"\n{10*'-'} Testing Different Layer Configs {10*'-'}")
    single_key_provenance_refactored(results_dir)
