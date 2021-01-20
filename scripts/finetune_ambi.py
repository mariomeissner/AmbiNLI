import argparse
import numpy as np
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from custom_bert import CustomBertForSequenceClassification
from datasets import Dataset, load_from_disk, concatenate_datasets

ambi_snli_path = "data/ambi-snli-tokenized"
ambi_mnli_path = "data/ambi-mnli-tokenized"
ambi_unli_path = "data/ambi-unli-tokenized"

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("output_dir")
parser.add_argument("--use_snli", action="store_true")
parser.add_argument("--use_mnli", action="store_true")
parser.add_argument("--use_unli", action="store_true")
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

dataset_list = []

if args.use_snli:
    dataset_list.append(load_from_disk(ambi_snli_path))
if args.use_mnli:
    dataset_list.append(load_from_disk(ambi_mnli_path))
if args.use_unli:
    dataset_list.append(load_from_disk(ambi_unli_path)["train"])
if not dataset_list:
    exit("Provide at least one dataset.")

dataset_list = [
    Dataset.from_dict(
        {
            "attention_mask": dataset["attention_mask"],
            "input_ids": dataset["input_ids"],
            "token_type_ids": dataset["token_type_ids"],
            "label": dataset["label"],
        }
    )
    for dataset in dataset_list
]

dataset = concatenate_datasets(dataset_list)
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = CustomBertForSequenceClassification.from_pretrained(args.model_path)

model.temperature = args.temperature

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    targets = np.argmax(p.label_ids, axis=1)
    return {"accuracy": (preds == targets).astype(np.float32).mean().item()}


training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    fp16=args.fp16,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,  # For padding collate
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
