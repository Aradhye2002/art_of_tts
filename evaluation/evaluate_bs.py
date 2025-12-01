import sys; sys.path.append("..")
from GLOBALS import *
import glob
from utils import read_json, read_cache, cache_output
from typing import Union
import numpy as np
import os

recompute_cache = True

# Only for evaluation of bs traces
# Gives the scores for num_beams=N

Answer = Union[int, str]

# Return np.array with x-axis representing method num_beams=N and
# y-axis representing metrics (acc, seq tokens, total tokens)
def get_per_model_dataset_sample(model: str, dataset: str, sample_id: int, N: int, prefix=".") -> np.ndarray:
    method = "bs"
    generic_path = f"results/{method}/{model}/{dataset}/{N}/{sample_id}_*.json"
    generic_path = os.path.join(prefix, generic_path)
    paths = sorted(glob.glob(generic_path))
    data = [read_json(path) for path in paths]
    # sanity check: all gt's should be same
    gt = data[0]["answer"]
    for x in data:
        assert(gt == x["answer"])
    data = [(int(x["completion_tokens"]), x["pred"]) for x in data]
    acc_output = np.zeros((1, 3))   # will return num_beams=N
    
    for l, pred in data:
        if pred == gt:
            acc_output[0, 0] += 1
        acc_output[0, 1] += l
        acc_output[0, 2] += N*l
    return acc_output/len(data)  # don't cache at this granularity

def get_per_model_dataset(model: str, dataset: str, N: int, prefix=".") -> np.ndarray:
    if not recompute_cache:
        return_output = read_cache(f"bs_{model}_{dataset}_{N}.pkl", prefix=prefix)
        return return_output
    acc_output = np.zeros((1, 3))
    num_samples = NUM_SAMPLES_MAP[dataset]
    for sample_id in range(num_samples):
        output = get_per_model_dataset_sample(model, dataset, sample_id, N, prefix=prefix)
        acc_output += output
    return_output = acc_output/num_samples
    cache_output(return_output, file_name=f"bs_{model}_{dataset}_{N}.pkl", prefix=".")
    return return_output

def get_per_model(model: str, N: int, prefix=".") -> np.ndarray:
    if not recompute_cache:
        return_output = read_cache(f"bs_{model}_{N}.pkl", prefix=prefix)
        return return_output
    acc_output = np.zeros((1, 3))
    for dataset in DATASETS:
        output = get_per_model_dataset(model, dataset, N, prefix=prefix)
        acc_output += output
    return_output = acc_output/len(DATASETS)
    cache_output(return_output, file_name=f"bs_{model}_{N}.pkl", prefix=".")
    return return_output

def get_per_dataset(dataset: str, N: int, reasoning="all", prefix=".") -> np.ndarray:
    if not recompute_cache:
        return_output = read_cache(f"bs_{dataset}_{N}.pkl", prefix=prefix)
        return return_output
    acc_output = np.zeros((1, 3))
    if reasoning == "all":
        models = ALL_MODELS
    elif reasoning == "reasoning":
        models = REASONING_MODELS
    else:
        models = NON_REASONING_MODELS
    if "Dapo-Qwen-32B" in models:
        models.remove("Dapo-Qwen-32B")
    for model in models:
        output = get_per_model_dataset(model, dataset, N, prefix=prefix)
        acc_output += output
    return_output = acc_output/len(models)
    cache_output(return_output, file_name=f"bs_{dataset}_{N}.pkl", prefix=".")
    return return_output

def get_overall(N: int, reasoning="all", prefix="."):
    if not recompute_cache:
        return_output = read_cache(f"bs_{N}.pkl", prefix=prefix)
        return return_output
    acc_output = np.zeros((1, 3))
    for dataset in DATASETS:
        output = get_per_dataset(dataset, N, reasoning=reasoning, prefix=prefix)
        acc_output += output
    return_output = acc_output/len(DATASETS)
    cache_output(return_output, file_name=f"bs_{N}.pkl", prefix=".")
    return return_output


for model in ALL_MODELS:
    if model == "Dapo-Qwen-32B":
        continue
    for N in range(2, 9):
        get_per_model(model, N, prefix="..")
        
for N in range(2, 9):
    get_overall(N, reasoning="all", prefix="..")