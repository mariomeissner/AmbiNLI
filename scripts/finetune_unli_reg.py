import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

import argparse
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from earlystopping import EarlyStoppingCallback
from logistic_bert import BertForSequenceLogistic
from datasets import load_from_disk
from sklearn.metrics import log_loss

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("dataset_path")
parser.add_argument("output_dir")
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--seed", default=666, type=int)
parser.add_argument("--nn", action="store_true")
args = parser.parse_args()

dataset = load_from_disk(args.dataset_path)
train_dataset = dataset['train']
dev_dataset = dataset['validation']
test_dataset = dataset['test']
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertForSequenceLogistic.from_pretrained(args.model_path)

if args.nn:
    torch.manual_seed(args.seed)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier.in_features, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 1)
    )
else:
    torch.manual_seed(args.seed)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)

for param in model.bert.parameters():
    param.requires_grad = False

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def bce_loss_numpy(logits, targets):
    eps = 1e-7
    return - np.mean(targets * np.log(eps + logits) + \
        (1 - targets) * np.log(1 - logits + eps))

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds =  sigmoid(preds.reshape((-1,)))
    targets = p.label_ids[:,0].reshape((-1,))
    return {"loss": float(bce_loss_numpy(preds, targets))}

callbacks_arr = [EarlyStoppingCallback(2)]

training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    learning_rate=args.lr,
    evaluation_strategy="epoch",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    fp16=True,
    metric_for_best_model='loss',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=callbacks_arr,
)

trainer.train()
trainer.save_model()

tmp = trainer.predict(test_dataset)
print(tmp)
test_probs, labels = tmp[0], tmp[1]
sigmoid = lambda x: 1 / (1 + np.exp(-x))
test_probs = sigmoid(test_probs)[:,0]
labels = labels[:,0]

spearman = spearmanr(test_probs, labels)
pearson = pearsonr(test_probs, labels)
mse = mean_squared_error(test_probs, labels)

print("--------------- Result on UNLI ---------------")
print(f"Spearman Corr: {spearman}")
print(f"Pearson Corr: {pearson}")
print(f"MSE Corr: {mse}")

print("---------------   End Result   ---------------")

# ~/../meissner/had_experiments/checkpoints/ambinli-subsets-comp/bb-ambismnli
# ~/../meissner/had_experiments/checkpoints/ambinli-subsets-comp/bb-ambismnli-gold
