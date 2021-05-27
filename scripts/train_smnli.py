import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from datasets import load_from_disk, concatenate_datasets, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("source_model_name")
parser.add_argument("output_dir")
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--fp32", default=False, action="store_true")
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

dataset_list = [
    load_from_disk("data/mnli-tokenized"),
    load_from_disk("data/snli-tokenized"),
]
train_dataset = concatenate_datasets(
    [
        Dataset.from_dict(
            {
                "attention_mask": dataset["train"]["attention_mask"],
                "input_ids": dataset["train"]["input_ids"],
                "token_type_ids": dataset["train"]["token_type_ids"],
                "label": dataset["train"]["label"],
            }
        )
        for dataset in dataset_list
    ]
)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# Prepare and train model
model = AutoModelForSequenceClassification.from_pretrained(
    args.source_model_name, return_dict=True, num_labels=3
)
tokenizer = AutoTokenizer.from_pretrained(args.source_model_name)

# Remove token-type-ids if roberta
if type(model) == RobertaForSequenceClassification:
    print("Removing token_type_ids because Roberta-chan doesn't like it.")
    train_dataset.remove_columns_("token_type_ids")

training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    fp16=not args.fp32,
    num_train_epochs=args.epochs,
    save_steps=0,
    seed=args.seed,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,  # For padded collate
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
