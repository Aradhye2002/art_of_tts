import sys; sys.path.append("../..")
from GLOBALS import ALL_MODELS
import numpy as np
import os
import pickle as pkl

DATASETS = ["gpqa", "aime2024", "aime2025-i", "aime2025-ii"]

def get_table(model):
    table = np.zeros((6, 4))
    # get seq and total BS counts
    if os.path.exists(f"../../evaluation/.cache/bs_{model}_8.pkl"):
        with open(f"../../evaluation/.cache/bs_{model}_8.pkl", "rb") as f:
            data = pkl.load(f)
        table[0, 0] = data[0, 1]; table[1, 0] = data[0, 2]
        for i, dataset in enumerate(DATASETS):
            # if overall bs cache exists assume individual model-dataset wise will also exist
            with open(f"../../evaluation/.cache/bs_{model}_{dataset}_8.pkl", "rb") as f:
                data = pkl.load(f)
            table[i+2, 0] = data[0, 0]
    
    # get vanilla data
    # cache should exist for all models
    with open(f"../../evaluation/.cache/{model}_8.pkl", "rb") as f:
        data = pkl.load(f)
        # MV
        table[:2, 1] = data[7, 1:]
        # LFS
        table[:2, 2] = data[8, 1:]
        # FFS
        table[:2, 3] = data[0, 1:]

    for i, dataset in enumerate(DATASETS):
        with open(f"../../evaluation/.cache/{model}_{dataset}_8.pkl", "rb") as f:
            data = pkl.load(f)
        # MV
        table[i+2, 1] = data[7, 0]
        # LFS
        table[i+2, 2] = data[8, 0]
        # FFS
        table[i+2, 3] = data[0, 0]
    return table

for model in ALL_MODELS:
    print(model)
    print(get_table(model), end="\n\n-------------\n")
    
