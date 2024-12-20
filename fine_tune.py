import torch
from datasets import load_dataset
from functools import partial


import torch
from datasets import load_dataset

from trl.trainer.utils import ConstantLengthDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import prepare_model_for_kbit_training

from tqdm import tqdm
import random
from diskcache import Index

# fix the seed
torch.manual_seed(42)
random.seed(42)


def _prompt(instruction, input_of_instruction):
    # print(row)
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:\n{instruction}"

    if len(input_of_instruction) > 2:
        prompt += f"### Input:\n{input_of_instruction}"

    prompt += ' \n\n### Response:\n'

    return prompt

def create_alpaca_prompt_with_response(example, eos_token):
    all_examples  = []
    for i in range(len(example['instruction'])):
        instruct = _prompt(example['instruction'][i], example['input'][i])
        prompt =  instruct + example["output"][i] +" "+  eos_token
        all_examples.append(prompt)
    return all_examples


def load_datasets(dname="lucasmccabe-lmi/CodeAlpaca-20k"):
    ds = load_dataset(dname, split="train")
    train_val, test = ds.train_test_split(test_size=0.10).values()
    train, val = train_val.train_test_split(test_size=0.2).values()

    # ratios of datasets pritn
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    return train, val, test, ds


def get_model_and_tokenizer(mname, special_tokens=None):
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(
        mname, torch_dtype=torch.bfloat16)

    if special_tokens is not None:
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=64,  # the rank of the LoRA matrices
        lora_alpha=16,  # the weight
        lora_dropout=0.1,  # dropout to add to the LoRA layers
        task_type="CAUSAL_LM",
        # the name of the layers to add LoRA
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def generat_hf(model, tokenizer, terminators,  prompt, max_new_tokens=256):
    encoding = tokenizer(prompt, return_tensors="pt",).to('cuda')
    model = model.cuda().eval()
    with torch.no_grad():
        outputs = model.generate(
            **encoding, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False)
        input_ids = encoding["input_ids"]
        response = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(response, skip_special_tokens=False)
    text = " ".join(text.split())
    print(f'Response (HF):\n ***|||{text}|||***\n\n')
    return text


def _manual_generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, terminators=None):
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

        # print(next_token_id.item(), eos_id)

        # print(f"Token: {next_token_id.item()}, terminator: {terminators}")  

        if next_token_id.item() in terminators:
            print(
                " =============================Found EOS token =============================")
            exit(0)
            break

        idx = torch.cat((idx, next_token_id), dim=-1)
    # print(f'Tokens {idx}')
    return idx


def generate_self(model, tokenizer, terminators, prompt, max_new_tokens=256, context_size=1024):
    encoding = tokenizer(prompt, return_tensors="pt",).to('cuda')
    model = model.cuda().eval()
    with torch.no_grad():
        m_outs = _manual_generate(
            model, encoding["input_ids"], max_new_tokens=max_new_tokens, context_size=context_size,  terminators=terminators)
        # print(m_outs.shape)
        m_outs = m_outs.squeeze(0)
        response = m_outs[encoding["input_ids"].shape[-1]:]
        # print(m_outs.shape)
    text = tokenizer.decode(response, skip_special_tokens=False)
    text = " ".join(text.split())
    print(f'Response (Manual):\n ***|||{text}|||***\n\n')
    return text


def main():
    cache_dir = "save_model_tokenizer/cache"
    eos_token = '<|waris_eos|>'

    train, val, test, whole_ds = load_datasets(dname='yahma/alpaca-cleaned')
    # for e in whole_ds:
    #     single_input = create_alpaca_prompt(e)
    #     break

    model, tokenizer = get_model_and_tokenizer(
        "gpt2", special_tokens=[eos_token])

    terminators = [tokenizer.eos_token_id,
                   tokenizer.convert_tokens_to_ids(eos_token)]

    # generat_hf(model, tokenizer, terminators,  single_input)
    # generate_self(model, tokenizer, terminators, single_input)

    instruction_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:"
    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
    
    
    training_args = SFTConfig(output_dir="./temp",  learning_rate=5e-5, num_train_epochs=10, per_device_train_batch_size=16,
                              do_eval=False, logging_steps=10, max_steps=-1, lr_scheduler_type='linear', packing=False, max_seq_length=1024, 
                              )

    

    trainer = SFTTrainer(
        model,
        train_dataset=train,
        eval_dataset=val,
        args=training_args,
        formatting_func=partial(
            create_alpaca_prompt_with_response, eos_token=eos_token),
        data_collator = collator

    )
    cache = Index(cache_dir)


    # trainer.train()
    # cache['model'] = model.cpu()
    # cache['tokenizer'] = tokenizer

    model = cache['model'].eval().cuda()
    # tokenizer = cache['tokenizer']

    for e in whole_ds:
        prompt = _prompt(e['instruction'], e['input'])
        print(f"Prompt: {prompt}")
        print(" ---------------- Response ----------------")
        generat_hf(model, tokenizer, terminators,  prompt)
        generate_self(model, tokenizer, terminators, prompt)
        # response end
        print(" ---------------- End ----------------")


main()
