#!/usr/bin/env bash
################
set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace
################

# Check number of arguments
if [ ! $# -eq 1 ]
  then
    echo "Usage: $0 model_name (including subdirectory names inside checkpoint folder)"
    echo "It assumes you run from project root directory."
    exit 0
fi

if [[ ! -d checkpoints/$1 ]]
then
    echo "Model $1 could not be found in checkpoints. Aborting."
    exit 1
fi

if [[ -d predictions/$1 ]] || [[ -d results/$1 ]]
then
    echo "Predictions and/or results already exist for this file."
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

mkdir -p predictions/$1 results/$1
python scripts/extract_nli_predictions.py checkpoints/$1/ snli data/snli_uids.json --temperature 1.0 > predictions/$1/snli.json
python scripts/extract_nli_predictions.py checkpoints/$1/ mnli data/mnli_uids.json --temperature 1.0 > predictions/$1/mnli.json
python ChaosNLI/src/scripts/evaluate.py --task_name uncertainty_nli --data_file ChaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl --prediction_file predictions/$1/snli.json > results/$1/snli.txt
python ChaosNLI/src/scripts/evaluate.py --task_name uncertainty_nli --data_file ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl --prediction_file predictions/$1/mnli.json > results/$1/mnli.txt