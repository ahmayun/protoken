from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
dataset = load_dataset("yelp_review_full")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(1000))


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=5)

# model = torch.compile(model)


training_args = TrainingArguments(output_dir="test_trainer")


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()


# del model
# del trainer
# torch.cuda.empty_cache()


# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# """### DataLoader

# Create a `DataLoader` for your training and test datasets so you can iterate over batches of data:
# """

# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# """Load your model with the number of expected labels:"""

# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# """### Optimizer and learning rate scheduler

# Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch:
# """

# from torch.optim import AdamW

# optimizer = AdamW(model.parameters(), lr=5e-5)

# """Create the default learning rate scheduler from [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer):"""

# from transformers import get_scheduler

# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )

# """Lastly, specify `device` to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes."""

# import torch

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# """<Tip>

# Get free access to a cloud GPU if you don't have one with a hosted notebook like [Colaboratory](https://colab.research.google.com/) or [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

# </Tip>

# Great, now you are ready to train! 🥳

# ### Training loop

# To keep track of your training progress, use the [tqdm](https://tqdm.github.io/) library to add a progress bar over the number of training steps:
# """

# from tqdm.auto import tqdm

# progress_bar = tqdm(range(num_training_steps))

# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

# """### Evaluate

# Just like how you added an evaluation function to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you'll accumulate all the batches with `add_batch` and calculate the metric at the very end.
# """

# import evaluate

# metric = evaluate.load("accuracy")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()

# """<a id='additional-resources'></a>

# ## Additional resources

# For more fine-tuning examples, refer to:

# - [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) includes scripts
#   to train common NLP tasks in PyTorch and TensorFlow.

# - [🤗 Transformers Notebooks](https://huggingface.co/docs/transformers/main/en/notebooks) contains various notebooks on how to fine-tune a model for specific tasks in PyTorch and TensorFlow.
# """
