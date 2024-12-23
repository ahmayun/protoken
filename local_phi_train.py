import sys
import logging


from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

from trl import SFTTrainer, SFTConfig, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer
import hydra
from functools import partial

################
# Prompt
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

    return {"train": train, "val": val, "test": test, "columns": columns, 'ds': ds}


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


def _manual_generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, terminators=None, tokenizer=None):
    model.eval()
    model = model.cuda()
    idx = idx.cuda()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            outputs = model(idx_cond)
            logits = outputs.logits

        logits = logits[:, -1, :]  # last token is the prediction

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # how does it get min val?
            logits = torch.where(
                logits < min_val, torch.tensor(float('-inf')), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        temp_id = next_token_id.item()

        if temp_id in terminators:
            print(terminators)
            print(
                f" =============================Found EOS token {temp_id} =============================")
            break
        idx = torch.cat((idx, next_token_id), dim=-1)
    return idx


def generate_self(model, tokenizer, terminators, prompt, max_new_tokens=256, context_size=1024):
    encoding = tokenizer(prompt, return_tensors="pt",).to('cuda')
    model = model.cuda().eval()
    with torch.no_grad():
        m_outs = _manual_generate(
            model, encoding["input_ids"], max_new_tokens=max_new_tokens, context_size=context_size,  terminators=terminators, tokenizer=tokenizer)
        m_outs = m_outs.squeeze(0)
        response = m_outs[encoding["input_ids"].shape[-1]:]
    text = tokenizer.decode(response, skip_special_tokens=False)
    text = " ".join(text.split())
    print(f'Response (Manual):\n ***|||{text}|||***\n\n')
    return text


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

    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id, 50256]

    for e in ds_dict['ds']:
        prompt = _prompt(e['instruction'], e['input'])
        print(" ---------------- Start ----------------")
        print(f"Prompt: {prompt}")
        # generat_hf(model, tokenizer, terminators,  prompt)
        generate_self(model, tokenizer, terminators, prompt)
        print(" ---------------- End ----------------")
        _ = input("Press Enter to continue")


main()
