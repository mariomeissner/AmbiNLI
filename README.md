# Embracing Ambiguity: Shifting the Training Target of NLI Models

## Paper and Citation

This is the code for the paper ["Embracing Ambiguity: Shifting the Training Target of NLI Models"](https://arxiv.org/abs/2106.03020). Please cite as the following:
```
@inproceedings{meissner-etal-2021-embracing,
    title = "Embracing Ambiguity: {S}hifting the Training Target of {NLI} Models",
    author = "Meissner, Johannes Mario  and
      Thumwanit, Napat  and
      Sugawara, Saku  and
      Aizawa, Akiko",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.109",
    doi = "10.18653/v1/2021.acl-short.109",
    pages = "862--869",
}
```

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

Set up the datasets (run once for each type of dataset you want to set up):
``` bash
python scripts/prepare_{*}.py
```

## Reproducing the main paper results

You can get results similar (sadly not equal due to small differences in seeds) to the paper by following these steps.

Pretrain a BERT model on 3 epochs of S+MNLI:
``` bash
python scripts/train_smnli.py bert-base-uncased checkpoints/base-models/bert-base-smnli
```

Finetune on some subset of AmbiNLI. The following example trains on SNLI + MNLI with ambiguity label distributions:
``` bash
python scripts/finetune_ambi.py checkpoints/base-models/bert-base-smnli/ checkpoints/ambinli-results/ambi-smnli --use_snli --use_mnli
```

Two or three epochs (`--epochs`) work best.

Run `python scripts/finetune_ambi.py --help` to see the remaining argument switches to run all the different experiments. Most importantly, run with `--use_gold_labels` to use gold-labels instead of the ambiguity distribution on whatever dataset(s) you selected.

Finally, you can evaluate the model performance on ChaosNLI through the following command:
``` bash
bash scripts/evaluate.sh checkpoints/ambinli-results/ambi-smnli bert
```

It will report the 4 metrics provided by ChaosNLI, on the SNLI and MNLI subsets. The result will also be recorded in the `results` folder.
...
