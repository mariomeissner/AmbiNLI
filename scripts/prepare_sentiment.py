import json
import datasets
from transformers import BertTokenizer

# Parameters
max_length = 128
num_procs = 16
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentiment_file = 'IMDB.csv'


def preprocess_function(example: dict):
    args = (example['review'],)
    return tokenizer(*args, max_length=max_length, truncation=True)

def label_preprocess_function(example: dict):
    return {'labels': float(example['sentiment']=='positive')}

unli_dataset = datasets.load_dataset(
    'csv', 
    data_files={
        'train': sentiment_file,
    }
)
unli_dataset = unli_dataset.map(preprocess_function, num_proc=num_procs)
unli_dataset = unli_dataset.map(label_preprocess_function, num_proc=num_procs)
unli_dataset = unli_dataset['train'].train_test_split(test_size=0.1, seed=666)
unli_dataset.save_to_disk("data/IMDB-tokenized")
