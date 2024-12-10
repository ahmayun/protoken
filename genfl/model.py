"""genfl: A Flower Baseline."""

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from functools import partial
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType,  get_peft_model_state_dict, set_peft_model_state_dict


# def data_collator(ds):
#     # print(ds)
#     labels = [f["labels"] for f in ds]
#     batch = {}
#     batch["input_ids"] = torch.stack([torch.tensor(f["input_ids"]) for f in ds]).cuda()
#     batch["attention_mask"] = torch.stack([torch.tensor(f["attention_mask"]) for f in ds]).cuda()
#     batch["labels"] = torch.tensor(labels).cuda()

#     print(batch['input_ids'].shape)


#     return batch


def initialize_model(mname, num_classes, peft):
    """Initialize the model with the given name."""
    model = AutoModelForSequenceClassification.from_pretrained(
        mname, num_labels=num_classes  # Ignore mismatched sizes
    )
    if peft:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            fan_in_fan_out=True,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def get_training_arguments(cfg, do_train=True, do_eval=False):
    return TrainingArguments(
        output_dir=cfg['dir'],
        learning_rate= cfg['lr'] if do_train else None,
        num_train_epochs=cfg['epochs'] if do_train else 0,
        eval_strategy='no' if not do_eval else 'epoch',
        do_train=do_train,
        do_eval=do_eval,
        fp16=True,
        disable_tqdm=True,  # Disable the progress bar
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
    training_args = get_training_arguments(
        train_cfg, do_train=True, do_eval=False)
    trainer = Trainer(model, training_args, train_dataset=train_cfg["train_data"],
                      eval_dataset=train_cfg["train_data"], compute_metrics=partial(compute_metrics, metric))
    trainer.train()
    eval_result = trainer.evaluate()
    model = model.cpu()
    return {'accuracy': eval_result['eval_accuracy'], 'loss': eval_result['eval_loss']}


def _hf_test(model, test_data, test_cfg):
    metric = evaluate.load("accuracy")
    training_args = get_training_arguments(test_cfg, do_train=False, do_eval=True)
    trainer = Trainer(model=model, args=training_args, eval_dataset=test_data,
                      compute_metrics=partial(compute_metrics, metric))
    eval_result = trainer.evaluate()
    model = model.cpu()
    return {'accuracy': eval_result['eval_accuracy'], 'loss': eval_result['eval_loss']}


def test(test_cfg, device=None):
    model, test_data = test_cfg['model'], test_cfg['test_data']
    return _hf_test(model, test_data, test_cfg)


def train(tconfig):
    """Train the neural network."""
    # return _hf_train(tconfig)
    return {'accuracy': -1, 'loss': -1}


def _get_state_dict(net, peft):
    if peft:
        state_dict = get_peft_model_state_dict(net)
    else:
        state_dict = net.state_dict()
    return state_dict


def get_parameters(model, peft):
    """Return model parameters as a list of NumPy ndarrays."""
    model = model.cpu()
    state_dict = _get_state_dict(model, peft)
    return [val.cpu().numpy() for _, val in state_dict.items()]


def set_parameters(net, parameters, peft):
    """Set model parameters from a list of NumPy ndarrays."""
    net = net.cpu()
    state_dict = _get_state_dict(net, peft)
    params_dict = zip(state_dict.keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    if peft:
        set_peft_model_state_dict(net, state_dict)
    else:
        net.load_state_dict(state_dict, strict=True)
