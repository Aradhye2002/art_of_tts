import glob
import sys; sys.path.append("../..")
from GLOBALS import *
import json
import pickle as pkl

difficulties = {}

for dataset in DATASETS:
    difficulties[dataset] = {}
    for sample_id in range(NUM_SAMPLES_MAP[dataset]):
        total_correct = 0
        total_output_len = 0
        total_num = 0
        for model in ALL_MODELS:
            regex_path = f"../../results/vanilla/{model}/{dataset}/{sample_id}_*.json"
            file_paths = glob.glob(regex_path)
            # calculate avg accuracy
            for file_path in file_paths:
                with open(file_path) as f:
                    data = json.load(f)
                if data["answer"] == data["pred"]:
                    total_correct += 1
                total_num += 1
                total_output_len += data["completion_tokens"]
        difficulties[dataset][sample_id] = (1 - total_correct/total_num, total_output_len/total_num)

with open("difficulties.pkl", "wb") as f:
    pkl.dump(difficulties, f)