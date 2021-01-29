#!/usr/bin/env bash
################
set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace
################

# Check number of arguments
if [ ! $# -eq 2 ]
  then
    echo "Usage: $0 model_path model_type"
    echo "It assumes you run from project root directory."
    exit 0
fi

if [[ ! -d $1 ]]
then
    echo "Model $1 could not be found. Aborting."
    exit 1
fi

model=${1#checkpoints/}

if [[ -d predictions/$model ]] || [[ -d results/$model ]]
then
    echo "Predictions and/or results already exist for this model."
    read -p "Overwrite? [y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "Cancelling..."
        exit 0
    fi
fi

if [[ -v PYTHONPATH ]]; then
    export PYTHONPATH=$PYTHONPATH:ChaosNLI/src:ChaosNLI/utest
else
    export PYTHONPATH=ChaosNLI/src:ChaosNLI/utest
fi

mkdir -p predictions/$model results/$model
python scripts/extract_nli_predictions.py checkpoints/$model/ $2 snli data/snli_uids.json > predictions/$model/snli.json
python scripts/extract_nli_predictions.py checkpoints/$model/ $2 mnli data/mnli_uids.json > predictions/$model/mnli.json
python ChaosNLI/src/scripts/evaluate.py --task_name uncertainty_nli --data_file ChaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl --prediction_file predictions/$model/snli.json > results/$model/snli.txt
python ChaosNLI/src/scripts/evaluate.py --task_name uncertainty_nli --data_file ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl --prediction_file predictions/$model/mnli.json > results/$model/mnli.txt
awk 'FNR == 6 {print "SNLI Divergence: "$2", Accuracy: "$5}' < results/$model/snli.txt
awk 'FNR == 6 {print "MNLI Divergence: "$2", Accuracy: "$5}' < results/$model/mnli.txt