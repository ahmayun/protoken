import torch
import math
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # Trainer,
    # TrainingArguments,
)
from typing import Dict
from collections import OrderedDict
from trl import SFTConfig, SFTTrainer, setup_chat_format
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from functools import partial


class ModelUtils:
    """Utility class for model parameter handling."""

    @staticmethod
    def _get_state_dict(net: torch.nn.Module, peft) -> Dict:
        if peft.enabled:
            return get_peft_model_state_dict(net)
        else:
            return net.state_dict()

    @staticmethod
    def get_parameters(model: torch.nn.Module, peft) -> list:
        """Return model parameters as a list of NumPy ndarrays."""
        model = model.cpu()
        state_dict = ModelUtils._get_state_dict(model, peft)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: list, peft) -> None:
        """Set model parameters from a list of NumPy ndarrays."""
        net = net.cpu()
        state_dict = ModelUtils._get_state_dict(net, peft)
        params_dict = zip(state_dict.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        if peft.enabled:
            set_peft_model_state_dict(net, state_dict)
        else:
            net.load_state_dict(state_dict, strict=True)


class PromptUtils:
    @staticmethod
    def test_prompt(instruction, input_of_instruction):
        prompt = f"### Instruction:\n{instruction}"
        if len(input_of_instruction) >= 1:
            prompt += f"\n### Input:\n{input_of_instruction}"
        prompt += '\n### Response:\n'
        return prompt

    @staticmethod
    def train_prompt(example, eos_token):
        instruct = PromptUtils.test_prompt(
            example['instruction'], example['input'])
        prompt = instruct + example["output"] + " " + eos_token + " "
        return prompt


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


def generate_response_from_llm(model, tokenizer, terminators, prompt, max_new_tokens=256, context_size=1024):
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


def get_model_and_tokenizer(model_cfg, peft):

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)

    if peft.config.task_type == 'CAUSAL_LM':
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name, **model_cfg.kwargs)
    elif peft.config.task_type == 'SEQ_CLS':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name, **model_cfg.kwargs)

    if 'microsoft/Phi-3-mini-4k-instruct' not in model_cfg.name:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    if peft.enabled:
        peft_conf = LoraConfig(**peft.config)
        model = get_peft_model(model, peft_conf)
        model.print_trainable_parameters()

    return model.cpu(), tokenizer


def train_or_eval_llm(model, tokenizer,  hf_ds, args_config, do_train, do_eval):
    train_conf = SFTConfig(do_train=do_train, do_eval=do_eval, **args_config)
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        train_dataset=hf_ds,
        eval_dataset=hf_ds,
        formatting_func=partial(PromptUtils.train_prompt, eos_token=tokenizer.eos_token))
    metrics = {}
    if do_train:
        train_result = trainer.train()
        temp_metrics = train_result.metrics
        trainer.log_metrics("train", temp_metrics)
        metrics['loss'] = temp_metrics['train_loss']

    if do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        metrics['loss'] = eval_result['eval_loss']

    metrics['perplexity'] = math.exp(metrics['loss'])

    trainer.log_metrics("Generic Metrics", metrics)
    metrics['accuracy'] = -1
    return metrics
