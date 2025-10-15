import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, get_peft_model
from collections import OrderedDict


def get_model_and_tokenizer(config):
    model_config = config["model_config"]

    model = AutoModelForCausalLM.from_pretrained(model_config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])

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


def evaluate_llm(model, tokenizer, eval_dataset):
    model.eval()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=eval_dataset.select(
            range(1)),  # dummy dataset to enable eval
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_eval_batch_size=32,
            seed=42,
            output_dir=None,
            report_to=None,
            do_train=False,
            do_eval=True,
            logging_strategy="no",
            save_strategy = "no",
            dataset_num_proc=4 
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
