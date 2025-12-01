REASONING_MODELS = ["GPT-OSS-120B", "Qwen3","QwQ", "R1", "R1DistilQwen", "Dapo-Qwen-32B"]
NON_REASONING_MODELS = ["Qwen3-235B", "Deepseek"]
ALL_MODELS = REASONING_MODELS + NON_REASONING_MODELS
DATASETS = ["aime2024", "aime2025-i", "aime2025-ii", "gpqa"]
NUM_SAMPLES_MAP = {"aime2024" : 30, "aime2025-i" : 15, "aime2025-ii" : 15, "gpqa" : 198}
# ((HF path, subname, split), type of answer)
DATASET_MAP = {
    "aime2024" : (("HuggingFaceH4/aime_2024", None, "train"), "subj"),
    "aime2025-i" : (("opencompass/AIME2025", "AIME2025-I", "test"), "subj"),
    "aime2025-ii" : (("opencompass/AIME2025", "AIME2025-II", "test"), "subj"),
    "math500" : (("HuggingFaceH4/MATH-500", None, "test"), "subj"),
    "gpqa" : (("aradhye/gpqa_diamond", None, "train"), "mcq"),
}
MODEL_MAP_API = {
    "r1" : "deepseek-ai/DeepSeek-R1",
    "chat" : "deepseek-ai/DeepSeek-V3",
    "qwq" : "Qwen/QwQ-32B",
    "r1-distill-qwen" : "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "qwen3" : "Qwen/Qwen3-32B",
    "qwen3-235b" : "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "gpt-oss-120b" : "openai/gpt-oss-120b",
}
MODEL_MAP_LOCAL = {
    "dapo-qwen-32b" : "BytedTsinghua-SIA/DAPO-Qwen-32B", 
    "dr_grpo-llama-3b" : "sail/Llama-3.2-3B-Oat-Zero", 
    "dr_grpo-qwen-7b" : "sail/Qwen2.5-Math-7B-Oat-Zero",
    "dr_grpo-qwen-1.5b" : "sail/Qwen2.5-Math-1.5B-Oat-Zero",
}
MODEL_NAME_MAP = {
    "r1" : "R1",
    "chat" : "Deepseek",
    "qwq" : "QwQ",
    "r1-distill-qwen" : "R1DistilQwen",
    "qwen3" : "Qwen3",
    "qwen3-235b" : "Qwen3-235B",
    "gpt-oss-120b" : "GPT-OSS-120B",
    "dapo-qwen-32b" : "DAPO-Qwen-32B", 
}
MODEL_TYPES=["Short-horizon", "Long-horizon", "Non-reasoning"]
MODEL_MAP_TYPE={
    "Short-horizon" : ["r1", "dapo-qwen-32b", "qwq"],
    "Long-horizon" : ["qwen3", "r1-distill-qwen", "gpt-oss-120b"],
    "Non-reasoning" : ["chat", "qwen3-235b"],
}