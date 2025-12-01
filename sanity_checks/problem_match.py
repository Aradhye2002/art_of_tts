import glob
import sys; sys.path.append("..")
from utils import read_json
from GLOBALS import ALL_MODELS, DATASETS, DATASET_MAP
from dataset import QuestionAnswerDataset

datasets = {name : QuestionAnswerDataset(*DATASET_MAP[name][0]) for name in DATASETS}

for method in ["bf", "bs", "vanilla"]:
    for model in ALL_MODELS:
        for dataset in DATASETS:
            file_paths = glob.glob(f"../results/{method}/{model}/{dataset}/*.json")
            for file_path in file_paths:
                problem_id = int(file_path.split("/")[-1].split("_")[0])
                problem1 = datasets[dataset][problem_id]["problem"]
                data = read_json(file_path)
                problem2 = data["problem"]
                answer1 = datasets[dataset][problem_id]["answer"]
                answer2 = data["answer"]
                if answer1 != answer2 or problem1 != problem2:
                    print(f"Failed for {file_path}")