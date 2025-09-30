from datasets import load_dataset


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


def get_client_dataset(cid: str, tokenizer, num_samples=100):
    G_DS = load_dataset(
        'waris-gill/llm-datasets-instruct-for-FL', split="train")

    chess_ds = G_DS.filter(
        lambda label: label == "chess",
        input_columns="label",
        batched=False,
        num_proc=8,
    )

    math_ds = G_DS.filter(
        lambda label: label == "math",
        input_columns="label",
        batched=False,
        num_proc=8,
    )

    dataset_map = {"0":  format_with_template(
        tokenizer, chess_ds), "1": format_with_template(tokenizer, math_ds)}
    return dataset_map.get(cid).select(range(num_samples))


def get_eval_datasets(tokenizer):
    chess_dataset = get_client_dataset("0", tokenizer)
    math_dataset = get_client_dataset("1", tokenizer)
    return {"chess": chess_dataset, "math": math_dataset}