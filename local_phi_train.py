import sys
import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import transformers
from trl import SFTTrainer, SFTConfig, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import hydra


##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example


def load_datasets(dname, tokenizer):
    ds = load_dataset(dname, split="train_sft")
    train_val, test = ds.train_test_split(test_size=0.10).values()
    train, val = train_val.train_test_split(test_size=0.2).values()
    # ratios of datasets pritn
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    column_names = list(train.features)

    processed_train_dataset = train.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_val_dataset = val.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to val_sft",
    )

    processed_test_dataset = test.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )

    # train = train.select(range(4*1000))
    return {"train": processed_train_dataset, "val": processed_val_dataset, "test": processed_test_dataset, "column_names": column_names}


def get_model_and_tokenizer(model_cfg, peft_cfg):
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name, **model_cfg.kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)

    if 'microsoft/Phi-3-mini-4k-instruct' not in model_cfg.name:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    peft_conf = LoraConfig(**peft_cfg)
    model = get_peft_model(model, peft_conf)
    model.print_trainable_parameters()
    return model, tokenizer


@hydra.main(config_path="./conf", config_name="train", version_base=None)
def main(cfg) -> None:
    ################
    # Model Loading
    ################
    model, tokenizer = get_model_and_tokenizer(
        cfg.model_config, cfg.peft_config)
    

    ds_dict = load_datasets("HuggingFaceH4/ultrachat_200k", tokenizer)

    # Training
    train_conf = SFTConfig(**cfg.train_config)
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["val"],
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)

    
main()
