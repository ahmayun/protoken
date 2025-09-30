import unsloth
import torch
from transformers import TextStreamer
from openai import OpenAI
import gc
from src.fl_train import get_client_dataset
from src.fl_prov import ProvTextGenerator, get_all_layers
from src.judge import llm_judge
from src.utils import CacheManager



def generate_response_with_provenance(model, tokenizer, dataset, sample_idx, client_models, terminators):
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

    result = ProvTextGenerator.generate_text(
        model, client_models, tokenizer, text, terminators
    )
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    match = llm_judge(
        predicted=result['response'], actual=conversation[2]['content'], client=client)

    print(f">> Actual Response (Ground Truth) -> {conversation[2]['content']}")
    print(f">> Generated Response: {result['response']}")
    print(
        f">> LLM Judge Verdict Generated Vs Actual Response: {'Match' if match else 'No Match'}")
    print(f">> Provenance Analysis: {result['client2part']}")
    print(
        f">> Predicted Client: {max(result['client2part'], key=result['client2part'].get)}")

    result['is_generation_correct'] = match

    return result


def evaluate_provenance(global_model, global_tokenizer, dataset, sample_idxs, client_models, terminators, expected_client_id):
    prov_acc = []
    for sample_idx in sample_idxs:
        result = generate_response_with_provenance(
            global_model, global_tokenizer, dataset, sample_idx,
            client_models, terminators
        )
        if result['is_generation_correct']:
            traced_client_id = max(
                result['client2part'], key=result['client2part'].get)
            prov_acc.append(traced_client_id == expected_client_id)
    return prov_acc


def rounds_provenance(rounds, sample_idxs):
    datasets = [
        ("chess", get_client_dataset("0"), "0"),
        ("math", get_client_dataset("1"), "1"),
    ]

    for round_num in rounds:

        print(f"\n{'='*60}")
        print(
            f"PROVENANCE ANALYSIS - ROUND {round_num}")
        print(f"{'='*60}")

        global_model, global_tokenizer, client_models = CacheManager.load_models_and_tokenizer_for_round(
            "Test", round_num)

        terminators = [global_tokenizer.eos_token_id]

        for dataset_name, dataset, expected_client_id in datasets:
            print(f"\n{'-'*40}")
            print(f"PROVENANCE - {dataset_name.upper()} DATASET")
            print(f"{'-'*40}")
            prov_acc_list = evaluate_provenance(global_model, global_tokenizer, dataset, sample_idxs,
                                                client_models, terminators, expected_client_id)
            print(
                f">> Provenance Accuracy: {sum(prov_acc_list)}/{len(prov_acc_list)} = {100.0 * sum(prov_acc_list)/len(prov_acc_list) if len(prov_acc_list) > 0 else 0.0:.2f}%")

        _ = [m.cpu() for m in client_models.values()]
        global_model.cpu()

        del client_models
        del global_model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    rounds_provenance(rounds=[0, 1], sample_idxs=list(range(100)))

    # test_all_available_rounds(sample_idx=10)
