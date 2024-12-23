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
from functools import partial

################
# Prompt Utils
################

def _prompt(instruction, input_of_instruction):
    prompt = f"### Instruction:\n{instruction}"
    if len(input_of_instruction) > 2:
        prompt += f"\n### Input:\n{input_of_instruction}"
    prompt += '\n### Response:\n'
    return prompt


def create_alpaca_prompt_with_response(example, eos_token):
    instruct = _prompt(example['instruction'], example['input'])
    prompt = instruct + example["output"] + " " + eos_token + " "
    return prompt


##################
# Data Processing
##################

def load_datasets(dname):
    ds = load_dataset(dname, split="train")
    train_val, test = ds.train_test_split(test_size=0.10).values()
    train, val = train_val.train_test_split(test_size=0.2).values()

    # ratios of datasets pritn
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    print(f"Columns: {list(train.features)}")

    columns = list(train.features)

    return {"train": train, "val": val, "test": test, "columns": columns}


################
# Model Loading
################

def get_model_and_tokenizer(model_cfg, peft_cfg):
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name, **model_cfg.kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)

    if 'microsoft/Phi-3-mini-4k-instruct' not in model_cfg.name:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # if model_cfg.name == 'microsoft/Phi-3-mini-4k-instruct':
    #     tokenizer.model_max_length = 2048
    #     # use unk rather than eos token to prevent endless generation
    #     tokenizer.pad_token = tokenizer.unk_token
    #     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
    #         tokenizer.pad_token)
    #     tokenizer.padding_side = 'right'

    peft_conf = LoraConfig(**peft_cfg)
    model = get_peft_model(model, peft_conf)
    model.print_trainable_parameters()
    return model, tokenizer


################
# Main
################
@hydra.main(config_path="./conf", config_name="train", version_base=None)
def main(cfg) -> None:
    ################
    # Model Loading
    ################
    model, tokenizer = get_model_and_tokenizer(
        cfg.model_config, cfg.peft_config)

    ds_dict = load_datasets('yahma/alpaca-cleaned')

    # Training
    train_conf = SFTConfig(**cfg.train_config)
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["val"],
        formatting_func=partial(
            create_alpaca_prompt_with_response, eos_token=tokenizer.eos_token),
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)


main()
