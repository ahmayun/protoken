import torch
from openai import OpenAI
import gc
import json
from pathlib import Path
from datetime import datetime
from src.utils.datasets import get_datasets_dict
from src.provenance.fl_prov import ProvTextGenerator, get_all_layers
from src.utils.judge import llm_judge
from src.utils.utils import CacheManager, get_model_and_tokenizer



def generate_response_with_provenance(model, tokenizer, dataset, sample_idx, client_models):
    print(
        f"\n\n{15*'-'} Input Index {sample_idx} Response Generation Provenance {15*'-'}")
    conversation = dataset['conversations'][sample_idx]

    messages = [
        {'role': conversation[0]['role'],
            'content': conversation[0]['content']},
        {"role": conversation[1]['role'],
            'content': conversation[1]['content']}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')
    
    print(f"{text}")

    result = ProvTextGenerator.generate_text(model, client_models, tokenizer, text)
    match = True
    # client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    # match = llm_judge(
    #     predicted=result['response'], actual=conversation[2]['content'], client=client)

    print(f"\n==== Generated Response===\n{result['response']}")
    print(f"\n**** Actual Response (Ground Truth) ****\n{conversation[2]['content']}")

    # print(
    #     f">> LLM Judge Verdict Generated Vs Actual Response: {'Match' if match else 'No Match'}")
    print(f">> Provenance Analysis: {result['client2part']}")
    # print(
    #     f">> Predicted Client: {max(result['client2part'], key=result['client2part'].get)}")

    result['is_generation_correct'] = match
    # _ = input("Press Enter to continue...")

    return result


def evaluate_provenance(global_model, global_tokenizer, dataset, sample_idxs, client_models, expected_client_id):
    prov_acc = []
    for sample_idx in sample_idxs:
        result = generate_response_with_provenance(
            global_model, global_tokenizer, dataset, sample_idx,
            client_models
        )
        if result['is_generation_correct']:
            traced_client_id = max(
                result['client2part'], key=result['client2part'].get)
            prov_acc.append(traced_client_id == expected_client_id)
    return prov_acc


def rounds_provenance(exp_key):
    experiment_config = CacheManager.load_experiment_configuration(exp_key)

    rounds, sample_idxs = range(experiment_config['fl']['num_rounds']), list(range(10))

    dataset_dict =  get_datasets_dict(experiment_config['dataset'])['test']
    _, global_tokenizer = get_model_and_tokenizer(experiment_config)
    print(f"tokenizer.eos_token_id: {global_tokenizer.eos_token_id}")
        # corresponding token str and id
    print(f"tokenizer.eos_token: {global_tokenizer.decode(global_tokenizer.eos_token_id)}")
    print(f"all special tokens: {global_tokenizer.special_tokens_map}")

    provenance_data = {}

    for round_num in rounds:

        print(f"\n{'='*60}")
        print(
            f"PROVENANCE ANALYSIS - ROUND {round_num}")
        print(f"{'='*60}")

        global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(
            exp_key, round_num)

        
    
        if len(client_models) == 0 and round_num == 0:
            print("No client models found for this round. Skipping provenance analysis.")
            continue
        elif len(client_models) == 0:
            raise ValueError("No client models found for this round. Cannot proceed with provenance analysis.")


        round_data = {}
        for expected_client_id, dataset in dataset_dict.items():
            dataset_name = experiment_config['dataset'][f'client_{expected_client_id}_dataset']

            print(f"\n{'-'*40}")
            print(f"PROVENANCE - {dataset_name.upper()} DATASET")
            print(f"{'-'*40}")
            prov_acc_list = evaluate_provenance(global_model, global_tokenizer, dataset, sample_idxs,
                                                client_models, expected_client_id)
            accuracy = 100.0 * \
                sum(prov_acc_list) / \
                len(prov_acc_list) if len(prov_acc_list) > 0 else 0.0
            round_data[dataset_name] = accuracy
            print(
                f">> Provenance Accuracy: {sum(prov_acc_list)}/{len(prov_acc_list)} = {accuracy:.2f}%")

        provenance_data[round_num] = round_data

        _ = [m.cpu() for m in client_models.values()]
        global_model.cpu()

        del client_models
        del global_model
        torch.cuda.empty_cache()
        gc.collect()

    return {
        "metadata": {
            "experiment_key": exp_key,
            "experiment_config": experiment_config,
            "timestamp": datetime.now().isoformat(),
            "total_rounds": len(rounds),
            "sample_count": len(sample_idxs),
        },
        "provenance_data": provenance_data
    }


        
