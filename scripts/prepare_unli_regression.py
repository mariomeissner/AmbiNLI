import json
import datasets
from transformers import BertTokenizer

# Parameters
max_length = 128
num_procs = 16
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
unli_path = 'data/unli/'


def preprocess_function(example: dict):
    args = (example["pre"], example["hyp"])
    return tokenizer(*args, max_length=max_length, truncation=True)

def label_preprocess_function(example: dict):
    return {'labels':[example['unli'], example['nli']]}

unli_dataset = datasets.load_dataset(
    'csv', 
    data_files={
        'train': unli_path + 'train.csv',
        'validation': unli_path + 'dev.csv',
        'test': unli_path + 'test.csv'
    }
)
unli_dataset = unli_dataset.map(preprocess_function, num_proc=num_procs)
unli_dataset = unli_dataset.map(label_preprocess_function, num_proc=num_procs)
unli_dataset.save_to_disk("data/unli-tokenized")
