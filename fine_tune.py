import torch
from datasets import load_dataset
from functools import partial


import torch
from datasets import load_dataset

from trl.trainer.utils import ConstantLengthDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, setup_chat_format

from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import prepare_model_for_kbit_training

from tqdm import tqdm
import random
from diskcache import Index

# fix the seed
torch.manual_seed(42)
random.seed(42)


def _prompt(instruction, input_of_instruction):
    prompt = f"### Instruction:\n{instruction}"
    if len(input_of_instruction) > 2:
        prompt += f"\n### Input:\n{input_of_instruction}"
    prompt += '\n### Response:\n'
    return prompt


def create_alpaca_prompt_with_response_collate(example, eos_token):
    all_examples = []
    for i in range(len(example['instruction'])):
        instruct = _prompt(example['instruction'][i], example['input'][i])
        prompt = instruct + example["output"][i] + " " + eos_token + " " 
        all_examples.append(prompt)
    return all_examples


def create_alpaca_prompt_with_response(example, eos_token):
    instruct = _prompt(example['instruction'], example['input'])
    prompt = instruct + example["output"] + " " + eos_token + " " 
    return prompt


def load_datasets(dname="lucasmccabe-lmi/CodeAlpaca-20k"):
    ds = load_dataset(dname, split="train")
    train_val, test = ds.train_test_split(test_size=0.10).values()
    train, val = train_val.train_test_split(test_size=0.2).values()

    # ratios of datasets pritn
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")

    # train = train.select(range(4*1000))


    return train, val, test, ds


def get_model_and_tokenizer(mname):
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(
        mname, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        mname, torch_dtype=torch.bfloat16)

    
    # model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,  # the rank of the LoRA matrices
        lora_alpha=8,  # the weight
        lora_dropout=0.1,  # dropout to add to the LoRA layers
        task_type="CAUSAL_LM",
        # the name of the layers to add LoRA
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # target_modules='all-linear',
    )
    model, tokenizer =  setup_chat_format(model=model, tokenizer=tokenizer)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
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
        
        
        # token =  tokenizer.decode(next_token_id.item())
        # print((token, next_token_id.item()))

       

        temp_id = next_token_id.item()

        if temp_id in terminators:
            print(terminators)
            print(
                f" =============================Found EOS token {temp_id} =============================")
            # exit(0)
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
    # print(f"ids: {response}")   
    text = tokenizer.decode(response, skip_special_tokens=False)
    text = " ".join(text.split())
    print(f'Response (Manual):\n ***|||{text}|||***\n\n')
    return text


def main():
    cache_dir = "save_model_tokenizer/cache"
    mname = 'gpt2'

    train, val, test, whole_ds = load_datasets(dname='yahma/alpaca-cleaned')

    model, tokenizer = get_model_and_tokenizer(mname)
    
   



    instruction_template = "### Instruction:\n"
    response_template = "\n### Response:\n"
    # collator = DataCollatorForCompletionOnlyLM(
    #     instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    training_args = SFTConfig(output_dir="./temp",  learning_rate=0.001, num_train_epochs=10, per_device_train_batch_size=16,
                              do_eval=False, logging_steps=10, max_steps=-1, lr_scheduler_type='linear', packing=True, max_seq_length=1024,
                              )


    print(f"Tokenizer eos: {tokenizer.eos_token_id}, eos_token: {tokenizer.eos_token}")

    trainer = SFTTrainer(
        model,
        train_dataset=train,
        eval_dataset=val,
        args=training_args,
        formatting_func=partial(
            create_alpaca_prompt_with_response, eos_token=tokenizer.eos_token),
        # data_collator=collator
    )
    cache = Index(cache_dir)

    # trainer.train()

    # cache['model'] = model.cpu()
    # cache['tokenizer'] = tokenizer

    # model = cache['model'].eval().cuda()
    # tokenizer = cache['tokenizer']

    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id, 50256]
    

    for e in whole_ds:
        prompt = _prompt(e['instruction'], e['input'])
        print(" ---------------- Start ----------------")
        print(f"Prompt: {prompt}")
        # generat_hf(model, tokenizer, terminators,  prompt)
        generate_self(model, tokenizer, terminators, prompt)
        print(" ---------------- End ----------------")
        _ = input("Press Enter to continue")


main()
