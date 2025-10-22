import torch
from tqdm import tqdm
from src.utils.judge import llm_judge


import logging
logger = logging.getLogger("Prov")


def _get_next_token_id(model, idx_cond):    
    outputs = model(idx_cond)
    logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token_id

def generate_text(model,  tokenizer, prompt,  max_new_tokens=64, context_size=2048):
    terminal_ids = [tokenizer.eos_token_id]

    try:
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        terminal_ids.append(end_of_turn_id)
    except:
        print("double check if <end_of_turn> token exists in the tokenizer")

    encoding = tokenizer(prompt, return_tensors="pt").to('cuda')
    idx = encoding["input_ids"]
    model.eval()    

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            token_dict = _get_next_token_id(model, idx_cond)
            next_token_id = token_dict["next_token_id"]
            temp_id = next_token_id.item()
            if temp_id in terminal_ids:
                break
            idx = torch.cat((idx, next_token_id), dim=-1)

    response = idx.squeeze(0)[encoding["input_ids"].shape[-1]:]
    text = tokenizer.decode(response, skip_special_tokens=False)
    text = " ".join(text.split())
    return text


def prepare_prompt(conversation, tokenizer):
    messages = [
        {'role': conversation[0]['role'],
            'content': conversation[0]['content']},
        # {'role': conversation[1]['role'],
        #     'content': conversation[1]['content']}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')

    return text



def find_inputs_ids_where_response_is_correct(model, tokenizer, label2dataset):
    new_ds_dict = {}
    for label, dataset in label2dataset.items():
        idx_where_response_is_correct = []
        counter = 0
        for i in tqdm(range(100), desc=f'Finding correct responses for label {label}'):
            conversation = dataset['messages'][i]
            prompt = prepare_prompt(conversation, tokenizer)
            

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            outputs = model.generate(
                input_ids=inputs['input_ids'], max_new_tokens=125)

            prompt_length = inputs['input_ids'].shape[1]

            generated_response = tokenizer.decode(
                outputs[0][prompt_length:], skip_special_tokens=True)
            actual_response = conversation[-1]['content']

            is_correct = llm_judge(generated_response, actual_response)

            logger.debug(f"Generated Response: {generated_response}\n")
            logger.debug(f"Actual Response: {actual_response}\n\n")

            if is_correct:
                logger.info(
                    f"\n\nLabel: {label}, Input Id: {i}, Judge: {is_correct}") 
                
                idx_where_response_is_correct.append(i)
                counter += 1
                if counter >= 10:
                    break

        new_ds_dict[label] = dataset.select(idx_where_response_is_correct)
    return new_ds_dict