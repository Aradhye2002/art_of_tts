from datasets import load_dataset
from torch.utils.data import Dataset
from utils import isint

class QuestionAnswerDataset(Dataset):
    def __init__(self, dataset, name=None, split="train"):
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, name=name, split=split)
        self.data = dataset
        self.problem_colname = "problem" if "problem" in self.data.features else "question"
        self.answer_colname = "answer"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        problem = self.data[self.problem_colname][idx]
        answer = self.data[self.answer_colname][idx]
        # All answers which are integer-like strings should be stored as integers
        if answer.endswith("^\\circ") and isint(answer[:3]):
            # This check is required to remove "^\\circ" which appears in aime2025-ii problem 4's given answer
            answer = answer[:3]
        if isint(answer):
            # This check is required to ensure consistency, since we directly compare all quantities against the provided answer
            answer = int(answer)
        return {"problem" : problem, "answer" : answer}
    
if __name__ == "__main__":
    dataset = QuestionAnswerDataset("aradhye/gpqa_diamond")
    print(len(dataset))
    print(dataset[0])
    
    dataset = QuestionAnswerDataset("HuggingFaceH4/aime_2024")
    print(len(dataset))
    print(dataset[0])
    
    dataset = QuestionAnswerDataset("opencompass/AIME2025", name="AIME2025-I", split="test")
    print(len(dataset))
    print(dataset[0])
    
    dataset = QuestionAnswerDataset("opencompass/AIME2025", name="AIME2025-II", split="test")
    print(len(dataset))
    print(dataset[0])
    
    dataset = QuestionAnswerDataset("HuggingFaceH4/MATH-500", split="test")
    print(len(dataset))
    print(dataset[0])
    