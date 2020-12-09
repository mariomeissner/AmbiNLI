import numpy as np
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_from_disk
from torch.utils.data import TensorDataset

dataset = load_from_disk("snli-tokenized")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained("goldlabel-finetune")

model.eval()
preds = []
for input_ids, attention_mask, token_type_ids in zip(
    dataset["test"]["input_ids"],
    dataset["test"]["attention_mask"],
    dataset["test"]["token_type_ids"],
):
    preds.append(
        model(
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(token_type_ids),
        )
    )

accuracy = (preds == p.label_ids).astype(np.float32).mean().item()