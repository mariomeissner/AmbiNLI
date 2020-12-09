import csv
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer

# Parameters
max_length = 128
num_procs = 16

snli_dev = pd.read_csv(
    "data/glue_data/SNLI/dev.tsv",
    delimiter="\t",
    index_col="pairID",
    quoting=csv.QUOTE_NONE,
)
snli_test = pd.read_csv(
    "data/glue_data/SNLI/test.tsv",
    delimiter="\t",
    index_col="pairID",
    quoting=csv.QUOTE_NONE,
)
chaos_snli = datasets.load_dataset(
    "json", data_files="ChaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl"
)["train"]

dataset = pd.concat((snli_dev, snli_test))
# Remove those entries present in chaosNLI
dataset = dataset.drop(chaos_snli["uid"])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_function(examples: dict):
    args = (examples["sentence1"], examples["sentence2"])
    result = tokenizer(*args, max_length=max_length, truncation=True)
    return result


def compute_distribution(example):
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    counts = np.bincount(
        (
            mapping[example["label1"]],
            mapping[example["label2"]],
            mapping[example["label3"]],
            mapping[example["label4"]],
            mapping[example["label5"]],
        ),
        minlength=3
    )
    distribution = counts / 5
    return {"label": list(distribution)}


dataset = Dataset.from_pandas(dataset)
dataset = dataset.map(compute_distribution)
dataset = dataset.map(preprocess_function, batched=True)
dataset.save_to_disk("data/ambi-snli-tokenized")
