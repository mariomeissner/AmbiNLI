from transformers import BertTokenizer
from datasets import load_dataset

# Parameters
max_length = 128
num_procs = 16

# Prepare dataset
dataset = load_dataset("glue", "mnli")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_function(examples: dict):
    args = (examples["premise"], examples["hypothesis"])
    result = tokenizer(*args, max_length=max_length, truncation=True)
    return result


dataset = dataset.map(preprocess_function, batched=True, num_proc=num_procs)
dataset = dataset.filter(lambda sample: sample["label"] != -1)
dataset.shuffle()
dataset.save_to_disk("data/mnli-tokenized")
