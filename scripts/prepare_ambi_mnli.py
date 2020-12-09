import csv
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer

# Parameters
max_length = 128
num_procs = 16

mnli_dev_match = pd.read_csv(
    "data/glue_data/MNLI/dev_matched.tsv",
    delimiter="\t",
    index_col="pairID",
    quoting=csv.QUOTE_NONE,
)
mnli_dev_mismatch = pd.read_csv(
    "data/glue_data/MNLI/dev_mismatched.tsv",
    delimiter="\t",
    index_col="pairID",
    quoting=csv.QUOTE_NONE,
)
chaos_mnli = datasets.load_dataset(
    "json", data_files="ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"
)["train"]

dataset = pd.concat((mnli_dev_match, mnli_dev_mismatch))
# Remove those entries present in chaosNLI
dataset = dataset.drop(chaos_mnli["uid"])

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
        minlength=3,
    )
    distribution = counts / 5
    return {"label": list(distribution)}


dataset = Dataset.from_pandas(dataset)
dataset = dataset.map(compute_distribution)
dataset = dataset.map(preprocess_function, batched=True)
dataset.save_to_disk("data/ambi-mnli-tokenized")
