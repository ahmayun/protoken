"""genfl: A Flower Baseline."""

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from functools import partial


# def data_collator(ds):
#     # print(ds)
#     labels = [f["labels"] for f in ds]
#     batch = {}
#     batch["input_ids"] = torch.stack([torch.tensor(f["input_ids"]) for f in ds]).cuda()
#     batch["attention_mask"] = torch.stack([torch.tensor(f["attention_mask"]) for f in ds]).cuda()
#     batch["labels"] = torch.tensor(labels).cuda()

#     print(batch['input_ids'].shape)


#     return batch


def initialize_model(mname, num_classes):
    """Initialize the model with the given name."""
    model = AutoModelForSequenceClassification.from_pretrained(
        mname, num_labels=num_classes)
    return model.cpu()


def get_training_arguments(cfg, do_train=True, do_eval=False):
    return TrainingArguments(
        output_dir=cfg['dir'],
        num_train_epochs=cfg['epochs'] if do_train else 0,
        eval_strategy='no' if not do_eval else 'epoch',
        do_train=do_train,
        do_eval=do_eval,
        fp16=True,
        # logging_dir='./logs',
        # logging_steps=10,
    )


def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def _hf_train(train_cfg):
    metric = evaluate.load("accuracy")
    model = train_cfg["model"]
    training_args = get_training_arguments(train_cfg, do_train=True, do_eval=False)
    trainer = Trainer(model, training_args, train_dataset=train_cfg["train_data"], eval_dataset=train_cfg["train_data"], compute_metrics=partial(compute_metrics, metric))
    trainer.train()
    model = model.cpu()
    eval_result = trainer.evaluate()
    return {'accuracy': eval_result['eval_accuracy'],'loss': eval_result['eval_loss']}

def _hf_test(model, test_data, test_cfg):
    metric = evaluate.load("accuracy")
    training_args = get_training_arguments(test_cfg, do_train=False, do_eval=True)
    trainer = Trainer(model=model, args=training_args, eval_dataset=test_data, compute_metrics=partial(compute_metrics, metric))
    eval_result = trainer.evaluate()
    model = model.cpu()
    return {'accuracy': eval_result['eval_accuracy'], 'loss': eval_result['eval_loss']}


def test(test_cfg, device=None):
    model, test_data = test_cfg['model'], test_cfg['test_data'] 
    return _hf_test(model, test_data, test_cfg)


def train(tconfig):
    """Train the neural network."""    
    return _hf_train(tconfig)
