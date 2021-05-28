# Embracing Ambiguity: Shifting the Training Target of NLI Models

## Paper and Citation
...

## Initial setup
Install necessary libraries:
``` bash
pip install -f requirements.txt
```

Clone ChaosNLI and set it up:
```
git clone https://github.com/easonnie/ChaosNLI.git ChaosNLI
source ChaosNLI/setup.sh
bash ChaosNLI/scripts/download_data.sh
```

Set up the datasets:
```
python scripts/prepare_snli.py
python scripts/prepare_mnli.py
python scripts/prepare_ambi_snli.py
python scripts/prepare_ambi_mnli.py
python scripts/prepare_ambi_unli.py
python scripts/prepare_chaosnli_finetune.py
```

## Reproducing the paper results

Pretrain a BERT model on 3 epochs of S+MNLI:
``` bash
python scripts/train_smnli.py bert-base-uncased checkpoints/base-models/bert-base-smnli
```

Finetune on some subset of AmbiNLI. The following example trains on SNLI + MNLI with ambiguity label distributions:
``` bash
python scripts/finetune_ambi.py checkpoints/base-models/bert-base-smnli/ checkpoints/ambinli-results/ambi-smnli --use_snli --use_mnli
```

Run `python scripts/finetune_ambi.py --help` to see the remaining argument switches to run all the different experiments. Most importantly, run with `--use_gold_labels` to use gold-labels instead of the ambiguity distribution on whatever dataset(s) you selected.

Finally, you can evaluate the model performance on ChaosNLI through the following command:
``` bash
bash scripts/evaluate.sh checkpoints/ambinli-results/ambi-smnli bert
```

It will report the 4 metrics provided by ChaosNLI, on the SNLI and MNLI subsets. The result will also be recorded in the `results` folder.
...
