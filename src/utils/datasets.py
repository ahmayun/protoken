from datasets import load_dataset
import os
import multiprocessing
_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4


SAMPLES_PER_CLIENT = 100
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42
TEST_DATASET_SIZE = 100

_dataset_cache = None

def format_with_template(tokenizer, dataset):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True, num_proc=8)

def _initialize_dataset_chunks():
    global _dataset_cache
    if _dataset_cache is not None:
        return
    
    global_dataset = load_dataset('waris-gill/llm-datasets-instruct-for-FL', split="train")

    chess_dataset = global_dataset.filter(
        lambda label: label == "chess",
        input_columns="label",
        batched=False,
        num_proc=8,
    ).shuffle(seed=RANDOM_SEED)

    math_dataset = global_dataset.filter(
        lambda label: label == "math",
        input_columns="label",
        batched=False,
        num_proc=8,
    ).shuffle(seed=RANDOM_SEED)

    chess_train_size = int(len(chess_dataset) * TRAIN_TEST_SPLIT)
    math_train_size = int(len(math_dataset) * TRAIN_TEST_SPLIT)
    
    chess_train = chess_dataset.select(range(chess_train_size))
    chess_test = chess_dataset.select(range(chess_train_size, len(chess_dataset)))
    
    math_train = math_dataset.select(range(math_train_size))
    math_test = math_dataset.select(range(math_train_size, len(math_dataset)))

    chess_chunks = []
    for i in range(5):
        start_idx = i * SAMPLES_PER_CLIENT
        end_idx = min(start_idx + SAMPLES_PER_CLIENT, len(chess_train))
        if start_idx < len(chess_train):
            chess_chunks.append(chess_train.select(range(start_idx, end_idx)))
        else:
            chess_chunks.append(None)

    math_chunks = []
    for i in range(5):
        start_idx = i * SAMPLES_PER_CLIENT
        end_idx = min(start_idx + SAMPLES_PER_CLIENT, len(math_train))
        if start_idx < len(math_train):
            math_chunks.append(math_train.select(range(start_idx, end_idx)))
        else:
            math_chunks.append(None)

    _dataset_cache = {
        'chess_chunks': chess_chunks,
        'math_chunks': math_chunks,
        'chess_test': chess_test,
        'math_test': math_test
    }

def get_client_dataset(cid, tokenizer, num_samples=SAMPLES_PER_CLIENT):
    if isinstance(cid, str):
        cid = int(cid)
    
    _initialize_dataset_chunks()
    
    if 0 <= cid <= 4:
        chunk = _dataset_cache['chess_chunks'][cid]
    elif 5 <= cid <= 9:
        chunk = _dataset_cache['math_chunks'][cid - 5]
    else:
        raise ValueError(f"Client ID {cid} out of range (0-9)")
    
    if num_samples < len(chunk):
        chunk = chunk.select(range(num_samples))
    
    return format_with_template(tokenizer, chunk)

def get_eval_datasets(tokenizer):
    _initialize_dataset_chunks()
    
    return {
        "chess": format_with_template(tokenizer, _dataset_cache['chess_test'].select(range(TEST_DATASET_SIZE))),
        "math": format_with_template(tokenizer, _dataset_cache['math_test'].select(range(TEST_DATASET_SIZE)))
    }
