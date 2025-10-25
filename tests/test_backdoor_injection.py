
from src.dataset.datasets import get_datasets_dict
from src.fl.config import ConfigManager
import numpy as np
from src.fl.model import get_model_and_tokenizer
from src.utils.generate import prepare_prompt
config = ConfigManager.load_default_config()





num_clients = 10
classes_per_client = 1
config["fl"]["num_clients"] = num_clients
config["fl"]["clients_per_round"] = num_clients
config["dataset"]["classes_per_client"] = classes_per_client





config['dataset']['inject_backdoor'] = True
config['dataset']['backdoor_clients'] = ['0', '1', '2']
config['dataset']['partition_strategy'] = 'iid'
config['model_config']['model_name'] = "meta-llama/Llama-3.2-1B-Instruct"

datasets_dict_bd = get_datasets_dict(num_clients, **config["dataset"])
model, tokenizer = get_model_and_tokenizer(config)

c0_ds_bd  = datasets_dict_bd['train']['0']

for row in c0_ds_bd:
    print(row)
    # prompt =  prepare_prompt(row['messages'], tokenizer)
    prompt = tokenizer.apply_chat_template(
        row['messages'],
        tokenize=False,
        add_generation_prompt=False,
    ).removeprefix('<bos>')

    print(f"Prompt:\n{prompt}\n")
    break

