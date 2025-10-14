from pathlib import Path
import torch
from openai import OpenAI
import gc


from src.utils.datasets import get_datasets_dict
from src.utils.model import get_model_and_tokenizer
from src.provenance.fl_prov import ProvTextGenerator, get_all_layers

from src.utils.judge import llm_judge
from src.utils.utils import save_json, CacheManager
from src.utils.plotting import plot_provenance_accuracy


def generate_response_with_provenance(global_model, client_models, tokenizer, text):

    result = ProvTextGenerator.generate_text(
        global_model, client_models, tokenizer, text)
    match = True
    # client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    # match = llm_judge(
    #     predicted=result['response'], actual=conversation[2]['content'], client=client)

    print(f"\n==== Generated Response===\n{result['response']}")

    print(
        f"\n**** Actual Response (Ground Truth) ****\n{conversation[2]['content']}")

    print(f"\n>> Provenance Analysis: {result['client2part']}")

    result['is_generation_correct'] = match

    return result


def foo(conversation, tokenizer):
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
    print(f"\n{'-'*5} Generation Prompt {'-'*5}\n{text}")
    return text


def single_round_provenance(exp_key, experiment_config, round_num, dataset_dict, tokenizer, test_inputs_idxs):
    def evaluate_provenance():
        prov_acc = []
        for sample_idx in test_inputs_idxs:
            text = foo(dataset['messages'][sample_idx], tokenizer)
            result = generate_response_with_provenance(
                global_model, client_models, tokenizer, text)
            if result['is_generation_correct']:
                traced_client_id = max(
                    result['client2part'], key=result['client2part'].get)
                prov_acc.append(traced_client_id == expected_client_id)
        return prov_acc

    print(f"{10*'-'} Round {round_num} {10*'-'}")
    global_model, client_models = CacheManager.load_models_and_tokenizer_for_round(
        exp_key, round_num)
    ds2acc = {}
    for expected_client_id, dataset in dataset_dict.items():

        prov_acc_list = evaluate_provenance()

        dataset_name = experiment_config['dataset'][f'client_{expected_client_id}_dataset']
        ds2acc[dataset_name] = 100.0 * sum(prov_acc_list) / len(prov_acc_list)

    _ = [m.cpu() for m in client_models.values()]
    global_model.cpu()
    del client_models
    del global_model
    torch.cuda.empty_cache()
    gc.collect()
    return ds2acc


def rounds_provenance(exp_key):
    train_config = CacheManager.load_experiment_configuration(exp_key)
    temp_model, tokenizer = get_model_and_tokenizer(train_config)
    layers_for_prov = get_all_layers(temp_model)

    dataset_dict = get_datasets_dict(train_config['dataset'])['test']

    round2provenance = {}
    test_input_idxs = list(range(10))
    total_rounds = 3  # train_config['fl']['num_rounds']
    for round_num in range(1, total_rounds):
        round2provenance[round_num] = single_round_provenance(
            exp_key, train_config, round_num, dataset_dict, tokenizer, test_input_idxs)

    return {"provenance": round2provenance, 'metadata': {'training_config': train_config}}


def full_cache_provenance(results_dir):
    print(f"{10*'-'} Running Provenance Analysis {10*'-'}")
    for exp_key in CacheManager.get_completed_experiments_keys():
        json_path = results_dir / f"provenance_{exp_key}.json"
        if json_path.exists():
            print(f"\nProvenance info already exists: {json_path}")
            print(f'\n\n {10*"="}')
            continue

        print(f"Running provenance analysis for experiment key: {exp_key}")
        result = rounds_provenance(exp_key=exp_key)
        CacheManager.set_provenance_results(exp_key, result)
        result['training'] = CacheManager.load_training_metrics(
            exp_key=exp_key)
        save_json(result, json_path)
        plot_provenance_accuracy(json_path, results_dir=results_dir)


def single_key_provenance(results_dir):
    exp_key = "[google_gemma-3-270m-it][rounds16][epochs-2][clients2][C0-medical-C1finance][LoRA-r8-alpha8][New2]"
    json_path = results_dir / f"single_provenance_{exp_key}.json"
    prov_dict = rounds_provenance(exp_key=exp_key)
    CacheManager.set_provenance_results(exp_key, prov_dict)
    prov_dict['training'] = CacheManager.load_training_metrics(exp_key=exp_key)
    save_json(prov_dict, json_path)


if __name__ == "__main__":
    results_dir = Path("results")
    # print(f"{10*'-'} Completed Training Keys {10*'-'}")
    # for exp_key in CacheManager.get_completed_experiments_keys():
    #     print(f"- {exp_key}")

    # # while True:
    # main()
    # print(f">> Sleeping")
    # time.sleep(10)

    # ---
    single_key_provenance(results_dir)
