import unsloth
import torch
import math
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from typing import Dict
from collections import OrderedDict
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def get_model_and_tokenizer(config):
    model_config = config["model_config"]
    lora_config = config.get("lora_config", {})
    get_chat_template_name = config["chat_template"]

    model, tokenizer = FastModel.from_pretrained(**model_config)

    if lora_config.get("use_lora", False):
        lora_params = {k: v for k, v in lora_config.items() if k != "use_lora"}
        model = FastModel.get_peft_model(model, **lora_params)

    tokenizer = get_chat_template(
        tokenizer, chat_template=get_chat_template_name)
    return model, tokenizer


class ModelUtils:

    @staticmethod
    def get_parameters(model: torch.nn.Module) -> list:
        model = model.cpu()
        use_lora = hasattr(model, 'peft_config')
        if use_lora:
            state_dict = get_peft_model_state_dict(model)
        else:
            state_dict = model.state_dict()
        return [val.cpu().float().numpy() for _, val in state_dict.items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: list) -> None:
        net = net.cpu()
        use_lora = hasattr(net, 'peft_config')
        if use_lora:
            state_dict = get_peft_model_state_dict(net)
            params_dict = zip(state_dict.keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in params_dict})
            set_peft_model_state_dict(net, state_dict)
        else:
            state_dict = net.state_dict()
            params_dict = zip(state_dict.keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in params_dict})
            net.load_state_dict(state_dict, strict=False)


def train_llm(model, tokenizer, train_dataset, sft_config_args):
    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        # dataset_text_field="text",
        args=SFTConfig(**sft_config_args)
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_proc=int(8)
    )

    metrics = trainer.train().metrics
    return {"loss": metrics['train_loss'], "perplexity": math.exp(metrics['train_loss'])}


def evaluate_llm(model, tokenizer, eval_dataset):
    model.eval()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=eval_dataset.select(
            range(1)),  # dummy dataset to enable eval
        eval_dataset=eval_dataset,
        dataset_text_field="text",

        args=SFTConfig(
            per_device_eval_batch_size=32,
            seed=42,
            output_dir=None,
            report_to=None,
            do_train=False,
            do_eval=True,
            # evaluation_strategy="no",  # we’ll call evaluate() manually
            logging_strategy="no",
            eval_on_start=True,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_proc=8
    )

    metrics = trainer.evaluate()

    return {"loss": metrics['eval_loss'], "perplexity": math.exp(metrics['eval_loss'])}
