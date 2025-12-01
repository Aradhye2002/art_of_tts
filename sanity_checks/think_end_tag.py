import glob
import sys; sys.path.append("..")
from utils import read_json
from GLOBALS import ALL_MODELS, DATASETS

think_end_tag = "</think>"

think_end_present = {   # all are verified
    "Dapo-Qwen-32B" : False,
    "GPT-OSS-120B" : False,
    "Phi4Reasoning" : True,
    "Qwen3" : True,
    "QwQ" : True,
    "R1" : True,
    "R1DistilQwen" : True,
    "Qwen3-235B" : False,
    "Deepseek" : False
}
for method in ["bf", "bs", "vanilla"]:
    for model in ALL_MODELS:
        for dataset in DATASETS:
            file_paths = glob.glob(f"../results/{method}/{model}/{dataset}/*.json")
            think_end_is_present = think_end_present[model]
            for file_path in file_paths:
                data = read_json(file_path)
                if think_end_tag in data["output"]:
                    if think_end_is_present:
                        break
                    else:
                        print(f"Failed for {file_path}")