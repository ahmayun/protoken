from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from src.dataset.datasets import get_datasets_dict
from src.fl.model import get_model_and_tokenizer, train_llm, evaluate_llm
from src.config.default import get_default_config


config = get_default_config()

model, tokenizer = get_model_and_tokenizer(config)

if tokenizer.chat_template:
    print("✅ Chat template found!")
else:
    print("❌ No chat template found. You need to set one manually.")


datasets_dict = get_datasets_dict(config['dataset'])

ds = datasets_dict['train']['0']
ds = ds.rename_column("conversations", "messages")


def format_with_template(tokenizer, dataset):
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True, num_proc=8)



sample_conv = ds[0]['messages']
text =  tokenizer.apply_chat_template(sample_conv, tokenize=False)
print("Sample formatted text:", text)


metrics = train_llm(model, tokenizer, ds, config['sft_config_args'])


# trainer_args = SFTConfig(**config['sft_config_args'])

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=ds,
#     args=trainer_args,
#     processing_class=tokenizer,
# )

# metrics =  trainer.train()

print(f' Lora model {model}')

print("Training metrics:", metrics)

test_ds = datasets_dict['test']['0']
test_ds = test_ds.rename_column("conversations", "messages")

sample_conv = test_ds[0]['messages']
text =  tokenizer.apply_chat_template(sample_conv, tokenize=False, add_generation_prompt=True)
print("Sample test formatted text:", text)

metrics = evaluate_llm(model, tokenizer, test_ds)
print("Evaluation metrics:", metrics)
