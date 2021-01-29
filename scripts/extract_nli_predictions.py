import fire
import json
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from datasets import load_dataset
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, BertTokenizer, DistilBertTokenizer

BATCH_SIZE = 64
MAX_LEN = 128
LABEL_DICT = {0: "entailment", 1: "neutral", 2: "contradiction"}


def extract_predictions(
    model_path: str,
    model_type: str,
    task_name: str,
    uids_list_path: str,
    temperature: int = 1.0,
):
    if task_name == "snli":
        dataset = load_dataset("snli")["validation"]
        dataset = dataset.filter(lambda sample: sample["label"] != -1)
        uid_list = json.load(uid_list_file := open(uids_list_path))
        uid_list_file.close()
    elif task_name == "mnli":
        dataset = load_dataset("glue", "mnli")["validation_matched"]
        uid_list = json.load(uid_list_file := open(uids_list_path))
        uid_list_file.close()
    else:
        exit("Invalid task_name.")

    if model_type == "bert":

        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif model_type == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.eval()
    model.to("cuda")
    # snli_dataset = load_from_disk(snli_dataset_path)["validation"]
    # mnli_dataset = load_from_disk(mnli_dataset_path)["validation_matched"]

    results = {model_path: {}}

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):

        batch = dataset[i : i + BATCH_SIZE]

        batch_inputs = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        batch_inputs.to("cuda")
        output = model(**batch_inputs, return_dict=True)
        logits = output.logits.cpu().detach().numpy()
        predicted_label_ids = np.argmax(logits, axis=1)
        predicted_probs = softmax(logits / temperature, axis=1)

        for j in range(logits.shape[0]):
            uid = uid_list[str(i + j)]
            predicted_label = LABEL_DICT[predicted_label_ids[j]]
            results[model_path][uid] = {}
            results[model_path][uid]["uid"] = uid
            results[model_path][uid]["predicted_label"] = predicted_label
            probs = list(map(float, predicted_probs[j]))
            results[model_path][uid]["predicted_probabilities"] = probs

    print(json.dumps(results))


if __name__ == "__main__":
    fire.Fire(extract_predictions)
