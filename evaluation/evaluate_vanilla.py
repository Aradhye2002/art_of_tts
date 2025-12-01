import sys; sys.path.append("..")
import glob
from utils import read_json, read_cache, cache_output
from typing import List, Union, Tuple
import numpy as np
from itertools import combinations
from GLOBALS import *
import os

recompute_cache = True
    
# Only for evaluation of vanilla traces
# Gives the scores for FFS-1, FFS-2, ..., FFS-N, LFS-1, LFS-2, ..., LFS-N, SD
# FFS-k means taking shortest k samples with a prediction (i.e., parsed answer should not be NO_ANSWER) and performing MV on them
# LFS-k means taking longest k samples with a prediction (i.e., parsed answer should not be NO_ANSWER) and performing MV on them

Answer = Union[int, str]
LIMIT_N = 8

# Make sure all data points across models, datasets and 
# parallel methods (FFS-1, FFS-2, ..., FFS-N, LFS-1, LFS-2, ..., LFS-N, SD) have atleast N outputs
def run_experiment(data: List[Tuple[int, Answer]], gt: Answer, N: int) -> np.ndarray:
    assert(len(data) == N)  # sanity check
    
    tot_len = sum(x for x, _ in data)
    avg_len = tot_len/N
    output = np.zeros((2*N+1, 3))
    max_len = max(x[0] for x in data)

    # FFS-1 to FFS-N
    for k in range(1, N+1):
        # find the kth smallest valid length
        valid_lengths = []
        for l, pred_ in data:
            if pred_ != "NO_ANSWER":
                valid_lengths.append(l)
        valid_lengths = sorted(valid_lengths)
        cutoff = valid_lengths[k-1] if len(valid_lengths) >= k else max_len
        # do majority voting
        pred_dict = {}
        for l, pred_ in data:
            if pred_ != "NO_ANSWER" and l <= cutoff:
                if pred_ not in pred_dict:
                    pred_dict[pred_] = 0
                pred_dict[pred_] += 1
        # find the majority vote
        if pred_dict:
            maj_vote = max(pred_dict, key=lambda key: pred_dict[key])
        else:
            maj_vote = "NO_ANSWER"
        is_correct = maj_vote == gt
        if is_correct:
            output[k-1, 0] = 1
        else:
            output[k-1, 0] = 0
        output[k-1, 1] = cutoff
        output[k-1, 2] = sum(l for l, _ in data if l<=cutoff)
    
    # LFS-1 to LFS-N
    for k in range(1, N+1):
        # find the (N-k+1)th largest valid length
        valid_lengths = []
        for l, pred_ in data:
            if pred_ != "NO_ANSWER":
                valid_lengths.append(l)
        valid_lengths = sorted(valid_lengths, reverse=True)
        cutoff = valid_lengths[k-1] if len(valid_lengths) >= k else 0
        # do majority voting
        pred_dict = {}
        for l, pred_ in data:
            if pred_ != "NO_ANSWER" and l >= cutoff:
                if pred_ not in pred_dict:
                    pred_dict[pred_] = 0
                pred_dict[pred_] += 1
        # find the majority vote
        if pred_dict:
            maj_vote = max(pred_dict, key=lambda key: pred_dict[key])
        else:
            maj_vote = "NO_ANSWER"
        is_correct = maj_vote == gt
        if is_correct:
            output[N+k-1, 0] = 1
        else:
            output[N+k-1, 0] = 0
        output[N+k-1, 1] = max_len
        output[N+k-1, 2] = tot_len
        
    # SD
    num_correct_sd = 0
    for _, pred_ in data:
        if pred_ == gt:
            num_correct_sd += 1
    is_correct_sd = num_correct_sd/N
    output[2*N, 0] = is_correct_sd
    output[2*N, 1] = avg_len
    output[2*N, 2] = avg_len
    return output

# Return np.array with x-axis representing methods (FFS-1, FFS-2, ..., FFS-N, LFS, SD) and
# y-axis representing metrics (acc, seq tokens, total tokens)
def get_per_model_dataset_sample(model: str, dataset: str, sample_id: int, N: int, prefix=".") -> np.ndarray:
    method = "vanilla"
    generic_path = f"results/{method}/{model}/{dataset}/{sample_id}_*.json"
    generic_path = os.path.join(prefix, generic_path)
    paths = sorted(glob.glob(generic_path))[:LIMIT_N]
    assert(len(paths) >= N)
    data = [read_json(path) for path in paths]
    # sanity check: all gt's should be same
    gt = data[0]["answer"]
    for x in data:
        assert(gt == x["answer"])
    data = [(int(x["completion_tokens"]), x["pred"]) for x in data]
    acc_output = np.zeros((2*N+1, 3))   # will return FFS-1, FFS-2, ..., FFS-N, LFS, SD
    
    num_combinations = 0
    for combination in combinations(data, r=N):
        num_combinations += 1
        data_chosen = list(combination)
        output = run_experiment(data_chosen, gt, N)
        acc_output += output
    return acc_output/num_combinations  # don't cache at this granularity

def get_per_model_dataset(model: str, dataset: str, N: int, prefix=".") -> np.ndarray:
    if not recompute_cache:
        return_output = read_cache(f"{model}_{dataset}_{N}.pkl", prefix=prefix)
        return return_output
    acc_output = np.zeros((2*N+1, 3))
    num_samples = NUM_SAMPLES_MAP[dataset]
    for sample_id in range(num_samples):
        output = get_per_model_dataset_sample(model, dataset, sample_id, N, prefix=prefix)
        acc_output += output
    return_output = acc_output/num_samples
    cache_output(return_output, file_name=f"{model}_{dataset}_{N}.pkl", prefix=".")
    return return_output

def get_per_model(model: str, N: int, prefix=".") -> np.ndarray:
    if not recompute_cache:
        return_output = read_cache(f"{model}_{N}.pkl", prefix=prefix)
        return return_output
    acc_output = np.zeros((2*N+1, 3))
    for dataset in DATASETS:
        output = get_per_model_dataset(model, dataset, N, prefix=prefix)
        acc_output += output
    return_output = acc_output/len(DATASETS)
    cache_output(return_output, file_name=f"{model}_{N}.pkl", prefix=".")
    return return_output

def get_per_dataset(dataset: str, N: int, choose="all", prefix=".") -> np.ndarray:
    # if not recompute_cache:
    #     return_output = read_cache(f"{dataset}_{N}.pkl", prefix=prefix)
    #     return return_output
    acc_output = np.zeros((2*N+1, 3))
    if choose == "all":
        models = ALL_MODELS
    elif choose == "reasoning":
        models = REASONING_MODELS
    else:
        models = NON_REASONING_MODELS
    for model in models:
        output = get_per_model_dataset(model, dataset, N, prefix=prefix)
        acc_output += output
    return_output = acc_output/len(models)
    cache_output(return_output, file_name=f"{dataset}_{N}.pkl", prefix=".")
    return return_output

def get_overall(N: int, choose="all", prefix="."):
    # if not recompute_cache:
    #     return_output = read_cache(f"{N}.pkl", prefix=prefix)
        # return return_output
    acc_output = np.zeros((2*N+1, 3))
    for dataset in DATASETS:
        output = get_per_dataset(dataset, N, choose=choose, prefix=prefix)
        acc_output += output
    return_output = acc_output/len(DATASETS)
    cache_output(return_output, file_name=f"{N}.pkl", prefix=".")
    return return_output

for N in range(1, 9):
    get_overall(N, choose="all", prefix="..")