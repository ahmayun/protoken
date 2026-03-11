import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, get_peft_model
from collections import OrderedDict


# Lazy mappings: only model IDs (strings). No HuggingFace loading until a model is requested.
MODEL_NAME_TO_TOKENIZER_ID = {
    'google/gemma-3-1b-pt': 'google/gemma-3-1b-it',
    'google/gemma-3-270m': 'google/gemma-3-270m-it',
    'meta-llama/Llama-3.2-1B': 'meta-llama/Llama-3.2-1B-Instruct',
}

MODEL_NAME_TO_UNSLOTH_CHAT_TEMPLATE = {
    'google/gemma-3-270m-it': 'gemma-3',
}


def _get_tokenizer_for_model(model_name: str):
    """Load tokenizer only when this model is explicitly requested (avoids HF auth at import)."""
    if model_name in MODEL_NAME_TO_TOKENIZER_ID:
        return AutoTokenizer.from_pretrained(MODEL_NAME_TO_TOKENIZER_ID[model_name])
    raise ValueError(f"Tokenizer for model {model_name} not found. Add it to MODEL_NAME_TO_TOKENIZER_ID in fl/model.py.")


def _get_unsloth_model_and_tokenizer(config):
    import unsloth
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    if config['use_lora']:
        raise NotImplementedError("LoRA with Unsloth is not yet implemented.")

    model_config = config["model_config"]
    # Unsloth supports torch_dtype; default to float16 for memory
    if "torch_dtype" not in model_config:
        model_config = {**model_config, "torch_dtype": torch.float16}
    model, tokenizer = FastModel.from_pretrained(**model_config)

    chat_template = MODEL_NAME_TO_UNSLOTH_CHAT_TEMPLATE.get(model_config['model_name'])
    if chat_template is not None:
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    return model, tokenizer


def _get_torch_dtype(model_config):
    """Resolve torch dtype from config (e.g. 'float16', 'bfloat16', 'float32')."""
    dtype_str = model_config.get('torch_dtype', 'float16')
    if isinstance(dtype_str, str):
        return getattr(torch, dtype_str)
    return dtype_str


def get_model_and_tokenizer(config):

    if config.get('use_unsloth', False):
        return _get_unsloth_model_and_tokenizer(config)

    model_config = config["model_config"]
    torch_dtype = _get_torch_dtype(model_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        torch_dtype=torch_dtype,
    )
    if model_config['model_name'].endswith('-it') or model_config['model_name'].endswith('-Instruct'):
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    else:
        tokenizer = _get_tokenizer_for_model(model_config['model_name'])

    if config['use_lora']:
        model = get_peft_model(model, LoraConfig(**config['lora_config']))
        model.print_trainable_parameters()
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
        # Preserve model dtype (e.g. float16) to reduce memory and transfer size
        return [val.cpu().numpy() for _, val in state_dict.items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: list) -> None:
        net = net.cpu()
        use_lora = hasattr(net, 'peft_config')
        if use_lora:
            state_dict = get_peft_model_state_dict(net)
            # Cast to model dtype (e.g. float16) in case aggregation returned float32/64
            dtype = next(iter(state_dict.values())).dtype
            params_dict = zip(state_dict.keys(), parameters)
            state_dict = OrderedDict({
                k: torch.tensor(v, dtype=dtype) for k, v in params_dict
            })
            set_peft_model_state_dict(net, state_dict)
        else:
            state_dict = net.state_dict()
            dtype = next(iter(state_dict.values())).dtype
            params_dict = zip(state_dict.keys(), parameters)
            state_dict = OrderedDict({
                k: torch.tensor(v, dtype=dtype) for k, v in params_dict
            })
            net.load_state_dict(state_dict, strict=True)


def train_llm(model, tokenizer, train_dataset, sft_config_args):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**sft_config_args)
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part="<start_of_turn>user\n",
    #     response_part="<start_of_turn>model\n",
    #     num_proc=int(8)
    # )

    metrics = trainer.train().metrics
    metrics['loss'] = metrics['train_loss']
    del trainer
    model = model.to("cpu")
    torch.cuda.empty_cache()
    return metrics


def evaluate_llm(model, tokenizer, eval_dataset, per_device_eval_batch_size=4):
    """Eval with small batch size by default to avoid server-side OOM."""
    model.eval()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=eval_dataset.select(
            range(1)),  # dummy dataset to enable eval
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_eval_batch_size=per_device_eval_batch_size,
            seed=42,
            output_dir=None,
            report_to=None,
            do_train=False,
            do_eval=True,
            logging_strategy="no",
            save_strategy="no",
            dataset_num_proc=1,
        ),
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part="<start_of_turn>user\n",
    #     response_part="<start_of_turn>model\n",
    #     num_proc=8
    # )

    metrics = trainer.evaluate()
    metrics['loss'] = metrics['eval_loss']
    del trainer
    torch.cuda.empty_cache()
    return metrics
