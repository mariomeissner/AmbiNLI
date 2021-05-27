import datasets
from transformers import BertTokenizer

# Parameters
max_length = 128
num_procs = 16
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_function(example: dict):
    args = (example["example"]["premise"], example["example"]["hypothesis"])
    return tokenizer(*args, max_length=max_length, truncation=True)


chaos_snli = datasets.load_dataset(
    "json", data_files="ChaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl"
)["train"]
chaos_mnli = datasets.load_dataset(
    "json", data_files="ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"
)["train"]
chaos_snli = chaos_snli.map(preprocess_function, num_proc=num_procs)
chaos_mnli = chaos_mnli.map(preprocess_function, num_proc=num_procs)
chaos_snli.rename_column_("label_dist", "label")
chaos_mnli.rename_column_("label_dist", "label")
chaos_snli.save_to_disk("data/chaos-snli-tokenized")
chaos_mnli.save_to_disk("data/chaos-mnli-tokenized")
