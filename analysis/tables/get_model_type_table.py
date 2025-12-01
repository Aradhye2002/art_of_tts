import sys; sys.path.append("../..")
import pickle as pkl
from GLOBALS import DATASETS, ALL_MODELS, NUM_SAMPLES_MAP
import numpy as np
import glob
import json

with open("../difficulty/difficulties.pkl", "rb") as f:
    difficulties = pkl.load(f)

# find average difficulty
avg_diff = sum([sum(tuple(zip(*difficulties[dataset].values()))[0])/len(difficulties[dataset]) for dataset in DATASETS])/len(DATASETS)
for model in ALL_MODELS:
    avg_length = 0
    # find mean length
    for dataset in DATASETS:
        avg_length_ = 0
        for problem_id in range(NUM_SAMPLES_MAP[dataset]):
            avg_length__ = 0
            pattern = f"../../results/vanilla/{model}/{dataset}/{problem_id}_*.json"
            filenames = glob.glob(pattern)
            for filename in filenames:
                with open(filename, "r") as f:
                    data = json.load(f)
                avg_length__ += data["completion_tokens"]
            avg_length_ += avg_length__ / 8
        avg_length += avg_length_ / NUM_SAMPLES_MAP[dataset]
    avg_length /= len(DATASETS)

    curr_grid = np.zeros(4); cnt_grid = np.zeros(4)
    for dataset in DATASETS:
        curr_grid_ = np.zeros(4); cnt_grid_ = np.zeros(4)
        for problem_id in range(NUM_SAMPLES_MAP[dataset]):
            short_avg = 0; long_avg = 0; short_cnt = 0; long_cnt = 0;
            pattern = f"../../results/vanilla/{model}/{dataset}/{problem_id}_*.json"
            filenames = glob.glob(pattern)
            for filename in filenames:
                with open(filename, "r") as f:
                    data = json.load(f)
                if data["completion_tokens"] <= avg_length:
                    short_avg += int(data["answer"] == data["pred"]); short_cnt += 1
                else:
                    long_avg += int(data["answer"] == data["pred"]); long_cnt += 1
            if short_cnt and long_cnt:
                if difficulties[dataset][problem_id][0] <= avg_diff:
                    curr_grid_[0] += short_avg / short_cnt; curr_grid_[1] += long_avg / long_cnt
                    cnt_grid_[:2] += 1
                else:
                    curr_grid_[2] += short_avg / short_cnt; curr_grid_[3] += long_avg / long_cnt
                    cnt_grid_[2:] += 1
        curr_grid += curr_grid_/cnt_grid_
    print(model, curr_grid/len(DATASETS))