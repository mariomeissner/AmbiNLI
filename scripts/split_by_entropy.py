import json
import pandas as pd
import numpy as np

dataset = []
with open("ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl") as f:
    for line in f:
        line = json.loads(line)
        dataset.append(line)

dataset = pd.DataFrame(dataset)
counts, bin_edges = np.histogram(dataset.entropy, bins=3)

binned_sets = []
for i in range(0, len(bin_edges) - 1):
    binned_sets.append(
        dataset[(dataset.entropy > bin_edges[i]) & (dataset.entropy <= bin_edges[i + 1])]
    )

for i, bin_set in enumerate(binned_sets):
    records = bin_set.to_dict("records")
    with open(f"data/binned_entropy/mnli_{i}.jsonl", "w") as f:
        for line in records:
            line = json.dumps(line)
            f.write(line)
            f.write("\n")
