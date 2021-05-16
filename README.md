# Embracing Ambiguity: Shifting the Training Target of NLI Models

## Reproducing the paper results

Pretrain a BERT model on 3 epochs of S+MNLI:
``` bash
python scripts/train_smnli.py bert-base-uncased checkpoints/base-models/bert-base-smnli
```

Finetune on some subset of AmbiNLI (example SNLI + MNLI with ambiguity label distributions):
``` bash
python scripts/finetune_ambi.py checkpoints/base-models/bert-base-smnli/ checkpoints/ambinli-results/ambi-smnli --use_snli --use_mnli
```

Run `python scripts/finetune_ambi.py --help` to see the remaining argument switches to run all the different experiments. Most importantly, run with `--use_gold_labels` to use gold-labels instead of the ambiguity distribution on whatever dataset(s) you selected.

...
