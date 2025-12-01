import sys; sys.path.append("..")
import torch
import argparse
import os
from dataset import QuestionAnswerDataset
from utils import isint
import json
from utils import add_template, parse_output
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from GLOBALS import MODEL_MAP_LOCAL, DATASET_MAP

# PREFIX_MAP is needed because some tokenizers don't add the <think> tag with add_generation_prompt=True
PREFIX_MAP = {
    "dapo-qwen-32b" : "", 
    "dr_grpo-llama-3b" : "", 
    "dr_grpo-qwen-7b" : "",
    "dr_grpo-qwen-1.5b" : "",
}

with open("start_strings.txt", "r") as f:
    START_STRINGS = [line.strip() for line in f.readlines()]
    

def get_generation_and_store(args, file_path, sample_id, example, model, tokenizer, template_type):
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    decode_fn = DECODE_MAP[args.decode_method]
    try:
        content, total_tokens = decode_fn(
            add_template(example["problem"], template_type), 
            args.model, 
            model, 
            max_tokens, 
            temperature, 
            top_p,
            tokenizer,
            sample_id,
            args,
        )
    except Exception as e:
        print(f"Failed for {file_path}: {e}")
        return
    # storing part
    print(f"Succeeded for {file_path}")
    assert(not os.path.exists(file_path))    # Avoid data-loss, calling function needs to ensure this
    assert(os.path.exists(os.path.dirname(file_path)))   # parent directory should exist, again needs to be handled by the calling function
    payload = {
        "problem" : example["problem"],
        "answer" : example["answer"],   # answer is guaranteed to be an integer if it is integer-like
        "output" : content,
        "completion_tokens" : total_tokens,
        "pred" : parse_output(content, template_type)
    }
    with open(file_path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )

def completion_vanilla(
    problem, 
    model_name,
    model, 
    max_tokens, 
    temperature, 
    top_p,
    tokenizer,
    sample_id,
    args
):
    # need to manually apply the chat template
    if model_name == "dr_grpo-llama-3b":
        prompt = apply_r1_template(problem)
    elif model_name in ["dr_grpo-qwen-7b", "dr_grpo-qwen-1.5b"]:
        prompt = apply_qwen_math_template(problem)
    else:
        messages = [{"role": "user", "content": problem}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    content = START_STRINGS[sample_id]
    prefix = PREFIX_MAP[args.model]
    
    generation_kwargs = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_new_tokens" : max_tokens,
    }
    
    input = tokenizer(prompt+prefix+content, return_tensors="pt").to("cuda")
    
    output = model.generate(**input, **generation_kwargs)
    content = tokenizer.decode(output[0])
    total_tokens = output.size(1) - input.input_ids.size(1)
    
    return content, total_tokens

DECODE_MAP = {
    "vanilla" : completion_vanilla,
}

def main():
    parser = argparse.ArgumentParser()
    # currently only support these five datasets
    parser.add_argument('--dataset', type=str, choices=["aime2024", "aime2025-i", "aime2025-ii", "math500", "gpqa"], required=True)
    # currently only support these four models
    parser.add_argument('--model', type=str, choices=["dapo-qwen-32b", "dr_grpo-llama-3b", "dr_grpo-qwen-7b", "dr_grpo-qwen-1.5b"], required=True)
    parser.add_argument('--decode_method', type=str, choices=["vanilla"], required=True)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--dir_path', type=str, default=".")
    parser.add_argument('--num_samples', type=int, default=4)
    args = parser.parse_args()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model_name = MODEL_MAP_LOCAL[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Fill args if not specified
    if args.model == "dapo-qwen-32b":
        if args.temperature is None:
            args.temperature = 1.0
        if args.top_p is None:
            args.top_p = 0.7
        if args.max_tokens is None:
            args.max_tokens = 20480
    elif args.model in ["dr_grpo-llama-3b", "dr_grpo-qwen-7b", "dr_grpo-qwen-1.5b"]:
        if args.temperature is None:
            args.temperature = 0.0
        if args.top_p is None:
            args.top_p = 1.0
        if args.max_tokens is None:
            args.max_tokens = 3000
    else:
        raise NotImplementedError
    
    dataset = QuestionAnswerDataset(*DATASET_MAP[args.dataset][0])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP_LOCAL[args.model])
    template_type = DATASET_MAP[args.dataset][1]
    
    # check which samples are alreadly there
    num_examples = len(dataset)
    os.makedirs(args.dir_path, exist_ok=True)
    file_names = os.listdir(args.dir_path)
    
    # preprocessing
    store = {i:set() for i in range(num_examples)}
    for file_name in file_names:
        if "_" in file_name and file_name.endswith(".json"):
            pref, suff = file_name.split(".")[0].split("_")
            if isint(pref) and isint(suff):
                pref = int(pref)
                suff = int(suff)
                if pref < num_examples and suff < args.num_samples:
                    file_path = os.path.join(args.dir_path, file_name)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    assert(data["problem"] == dataset[pref]["problem"])
                    store[pref].add(suff)

    for i, example in enumerate(dataset):
        for j in range(args.num_samples):
            if j not in store[i]:
                file_path = os.path.join(args.dir_path, f"{i}_{j}.json")
                get_generation_and_store(args, file_path, j, example, model, tokenizer, template_type)

if __name__ == "__main__":
    main()