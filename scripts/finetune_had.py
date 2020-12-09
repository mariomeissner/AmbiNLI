import argparse
import numpy as np
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from custom_bert import CustomBertForSequenceClassification
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("train_dataset_path")
parser.add_argument("eval_dataset_path")
parser.add_argument("output_dir")
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--epochs", default=5, type=int)
args = parser.parse_args()

train_dataset = load_from_disk(args.train_dataset_path)
eval_dataset = load_from_disk(args.eval_dataset_path)
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = CustomBertForSequenceClassification.from_pretrained(args.model_path)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    targets = np.argmax(p.label_ids, axis=1)
    return {"accuracy": (preds == targets).astype(np.float32).mean().item()}


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
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model()
