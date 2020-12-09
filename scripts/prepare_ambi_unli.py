import datasets
from transformers import BertTokenizer

# Parameters
max_length = 128
num_procs = 16
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
unli_path = "data/unli/"


def preprocess_function(example: dict):
    args = (example["pre"], example["hyp"])
    return tokenizer(*args, max_length=max_length, truncation=True)


def label_preprocess_function(example: dict):
    if example["unli"] < 0.5:
        label_count = [0.0, 2 * example["unli"], 1 - 2 * example["unli"]]
    else:
        label_count = [2 * example["unli"] - 1, 2 * (1 - example["unli"]), 0.0]
    # assert sum(label_count) == 1
    return {"label": label_count}


unli_dataset = datasets.load_dataset(
    "csv",
    data_files={
        "train": unli_path + "train.csv",
        "validation": unli_path + "dev.csv",
        "test": unli_path + "test.csv",
    },
)
unli_dataset = unli_dataset.map(preprocess_function)
unli_dataset = unli_dataset.map(label_preprocess_function)
unli_dataset.save_to_disk("data/ambi-unli-tokenized")
