from datasets import load_dataset, concatenate_datasets, DatasetDict
import logging

logger = logging.getLogger("fl_ds")
logger.setLevel(logging.DEBUG)  # or logging.INFO as needed
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    logger.addHandler(handler)


# Dataset, Preprocessing, Chat Templates, etc
def dataset_adapter(name: str):
    name = name.lower()
    if name == "chess":
        hf_name = "Thytu/ChessInstruct"

        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": ex.get(
                        "task", "You are a helpful assistant.")},
                    {"role": "user", "content": ex.get("input")},
                    {"role": "assistant", "content": ex.get(
                        "expected_output")},
                ]
            }

        return hf_name, convert_to_chatml

    elif name == "math":
        hf_name = "m-gopichand/deepmind_math_dataset_processed"

        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": "You are a helpful assistant. Provide concise, correct solutions."},
                    {"role": "user", "content": ex.get("question")},
                    {"role": "assistant", "content": ex.get("answer")},
                ]
            }

        return hf_name, convert_to_chatml

    elif name == "medical":
        hf_name = "medalpaca/medical_meadow_medical_flashcards"
        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": "You are a helpful assistant. Provide concise, correct solutions."},
                    {"role": "user", "content": ex.get("input")},
                    {"role": "assistant", "content": ex.get("output")},
                ]
            }

        return hf_name, convert_to_chatml
    
    elif name == "coding":
        hf_name = "flwrlabs/code-alpaca-20k"
        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": "You are a helpful assistant. Provide concise, correct solutions."},
                    {"role": "user", "content": ex.get("instruction") + " " + ex.get("input", "") },
                    {"role": "assistant", "content": ex.get("output")},
                ]
            }
        return hf_name, convert_to_chatml
    
    elif name == "finance":
        hf_name = "flwrlabs/fingpt-sentiment-train"
        def convert_to_chatml(ex):
            return {
                "conversations": [
                    {"role": "system", "content": "You are a helpful assistant. Provide concise, correct solutions."},
                    {"role": "user", "content": ex.get("instruction") + " " + ex.get("input", "") },
                    {"role": "assistant", "content": ex.get("output")},
                ]
            }
        return hf_name, convert_to_chatml


    else:
        raise ValueError(f"Unknown dataset adapter: {name}")


def dataset_sizes_with_names(datasets: list[str]) -> dict[str, int]:
    sizes = {}
    for name in datasets:
        hf_name, _ = dataset_adapter(name)
        ds = load_dataset(hf_name, split="train")  # full train
        sizes[name] = len(ds)
    return sizes


def to_chatml_with_meta(name: str, min_size: int):
    hf_name, convert_to_chatml = dataset_adapter(name)
    ds = load_dataset(hf_name, split="train")  # full train

    # replicate your current filtering rule for math
    if name.lower() == "math":
        # ds = ds.filter(lambda ex: ex.get("difficulty") == "train-easy")
        ds = ds.filter(
            lambda difficulty: difficulty == "train-easy",
            input_columns="difficulty",
            batched=False,
            num_proc=8,
        )

    # downsample to min_size
    if len(ds) > min_size:
        logger.info(f"Downsampled {name} to {min_size} examples")
        ds = ds.shuffle(seed=42).select(range(min_size))

    ds = ds.map(convert_to_chatml, num_proc=8)
    ds = ds.add_column("label", [name] * len(ds))
    ds = ds.add_column("source", [hf_name] * len(ds))

    ds = ds.select_columns(["conversations", "label"])
    return ds


def build_train_only(repo_id: str, datasets: list[str]):
    dname2size = dataset_sizes_with_names(datasets)  # just to print sizes
    logger.info(f"Dataset sizes: {dname2size}")
    minimize_size = min(dname2size.values())
    logger.info(f"Min size: {minimize_size}")

    pieces = [to_chatml_with_meta(name, minimize_size) for name in datasets]
    train = concatenate_datasets(pieces)
    logger.info(f"Combined dataset size: {len(train)}")
    print(train[0])

    for k in train.column_names:
        logger.info(f"Column: {k}, example: {train[0][k]}")

    logger.info(f'train.column_names: {train.column_names}')

    # Push a single-split repo with just "train"
    DatasetDict({"train": train}).push_to_hub(repo_id, private=True)
    # print(f"Pushed combined train-only dataset to: {repo_id}")


def main():
    build_train_only(repo_id="llm-datasets-instruct-for-FL",
                     datasets=['chess', 'math', 'medical', 'coding', 'finance'])


if __name__ == "__main__":
    main()
