import sys; sys.path.append("..")
import argparse
import os
from openai import AsyncOpenAI
from dataset import QuestionAnswerDataset
import json
import asyncio
from utils import add_template, parse_output
from transformers import AutoTokenizer
from GLOBALS import MODEL_MAP_API, DATASET_MAP, MODEL_NAME_MAP

# Max number of times API calls will be made for one example
# For budget forcing this is the maximum number of times the WAIT_TOKEN will be added
MAX_ITERS = 100     
WAIT_TOKEN = "Wait"
MAX_ANSWER_TOKENS = 1000

# PREFIX_MAP is needed because some tokenizers don't add the <think> tag with add_generation_prompt=True
PREFIX_MAP = {
    "r1" : "",  # Checked
    "chat" : "",  # Checked (<think> tag not needed since non-reasoning model)
    "qwq" : "",  # Checked
    "r1-distill-qwen" : "",  # Checked
    "phi4" : "<think>",  # Checked
    "qwen3" : "<think>\n",  # Checked
    "qwen3-235b" : "",  # Checked (<think> tag not needed since non-reasoning model)
    "gpt-oss-120b" : "<|channel|>analysis<|message|>",  # Checked
}

with open("start_strings.txt", "r") as f:
    START_STRINGS = [line.strip() for line in f.readlines()]

async def get_generation_and_store(args, file_path, sample_id, example, openai_client, tokenizer, template_type, semaphore, model_id, decode_method, content=None):
    if os.path.exists(file_path):   # Don't do anything if the file already exists
        return
    model_name = MODEL_MAP_API[model_id]
    max_tokens = args["max_tokens"]
    temperature = args["temperature"]
    top_p = args["top_p"]
    decode_fn = DECODE_MAP[decode_method]
    try:
        async with semaphore:   # guard critical section
            content, total_tokens = await decode_fn(
                openai_client,
                add_template(example["problem"], template_type), 
                model_name, 
                max_tokens, 
                temperature, 
                top_p,
                tokenizer,
                sample_id,
                args,
                model_id,
                content,
            )
    except Exception as e:
        print(f"Failed for {file_path}: {e}", flush=True)
        return
    # storing part
    print(f"Succeeded for {file_path}", flush=True)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    assert(os.path.exists(os.path.dirname(file_path)))
    payload = {
        "problem" : example["problem"],
        "answer" : example["answer"],   # answer is guaranteed to be an integer if it is integer-like
        "output" : content,
        "completion_tokens" : total_tokens,
        "pred" : parse_output(content, template_type)
    }
    with open(file_path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)

async def completion_vanilla(
    openai_client, 
    problem, 
    model_name, 
    max_tokens, 
    temperature, 
    top_p,
    tokenizer,
    sample_id,
    args,
    model_id,
    content=None
):
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if content is None:
        content = START_STRINGS[sample_id]
    prefix = PREFIX_MAP[model_id]
    total_tokens = len(tokenizer.encode(content))
    iters = 0
    while (True):
        completion = await openai_client.completions.create(
            model=model_name,
            prompt=prompt+prefix+content,
            stream=False,
            n=1,
            max_tokens=max_tokens-total_tokens,
            temperature=temperature,
            top_p=top_p
        )
        content += completion.choices[0].text
        completion_tokens = completion.usage.completion_tokens
        total_tokens += completion_tokens
        iters += 1
        if total_tokens >= max_tokens:
            break
        if completion.choices[0].finish_reason == "stop":
            break
        if iters >= MAX_ITERS:
            break
    return content, total_tokens

async def completion_bf(
    openai_client, 
    problem, 
    model_name, 
    max_tokens, 
    temperature, 
    top_p,
    tokenizer,
    sample_id,
    args,
    model_id,
    content=None,
):
    assert max_tokens > MAX_ANSWER_TOKENS
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    content = START_STRINGS[sample_id]
    prefix = PREFIX_MAP[model_id]
    total_tokens = 0
    iters = 0
    while (True):
        completion = await openai_client.completions.create(
            model=model_name,
            prompt=prompt+prefix+content,
            stream=False,
            n=1,
            max_tokens=max_tokens-total_tokens-MAX_ANSWER_TOKENS,
            temperature=temperature,
            top_p=top_p,
            stop=args.stop_strings,
        )
        content += completion.choices[0].text
        total_tokens += completion.usage.completion_tokens
        iters += 1
        if total_tokens >= max_tokens-MAX_ANSWER_TOKENS:
            break
        if iters >= MAX_ITERS:
            break
        content += f"\n\n{WAIT_TOKEN}"
        total_tokens += 2   # Assuming WAIT_TOKEN is "Wait"
    # Give some tokens to produce the answer
    content += "**Final Answer**\n"
    total_tokens += 4
    completion = await openai_client.completions.create(
            model=model_name,
            prompt=prompt+prefix+content,
            stream=False,
            n=1,
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=temperature,
            top_p=top_p,
        )
    content += completion.choices[0].text
    total_tokens += completion.usage.completion_tokens
    return content, total_tokens

async def completion_bs(
    openai_client, 
    problem, 
    model_name, 
    max_tokens, 
    temperature, 
    top_p,
    tokenizer,
    sample_id,
    args,
    model_id,
    content=None,
):
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    content = START_STRINGS[sample_id]
    prefix = PREFIX_MAP[model_id]
    total_tokens = 0
    iters = 0
    while (True):
        completion = await openai_client.completions.create(
            model=model_name,
            prompt=prompt+prefix+content,
            stream=False,
            n=1,
            max_tokens=max_tokens-total_tokens,
            temperature=temperature,
            top_p=top_p,
            best_of=args["num_beams"]
        )
        content += completion.choices[0].text
        completion_tokens = completion.usage.completion_tokens
        total_tokens += completion_tokens
        if total_tokens >= max_tokens:
            break
        if completion.choices[0].finish_reason == "stop":
            break
        iters += 1
        if iters >= MAX_ITERS:
            break
    return content, total_tokens

DECODE_MAP = {
    "vanilla" : completion_vanilla,
    "bf" : completion_bf,
    "bs" : completion_bs,
}

async def main():
    import yaml
    parser = argparse.ArgumentParser()
    # currently only support these five datasets
    parser.add_argument('--datasets', type=str, choices=["aime2024", "aime2025-i", "aime2025-ii", "gpqa", "all"], nargs="+", required=True)
    # currently only support these seven models
    parser.add_argument('--models', type=str, choices=["r1", "chat", "qwq", "r1-distill-qwen", "qwen3", "qwen3-235b", "gpt-oss-120b", "all"], nargs="+", required=True)
    parser.add_argument('--decode_method', type=str, choices=["vanilla", "bf", "bs", "vanilla_error_correction"], required=True)    # Currently can select only one at a time
    parser.add_argument('--base_url', type=str, default="https://api.deepinfra.com/v1/openai")
    parser.add_argument('--bucket_size', type=int, default=200)
    main_args = parser.parse_args()
    tasks = []
    
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url=main_args.base_url,
        timeout=None
    )
    models = main_args.models
    datasets = main_args.datasets
    with open("config.yml", "r") as f:
        args = yaml.load(f, yaml.FullLoader)
        
    
    if "all" in models:
        assert(len(models) == 1)
        models = ["r1", "chat", "qwq", "r1-distill-qwen", "qwen3", "qwen3-235b", "gpt-oss-120b"]
    if "all" in datasets:
        assert(len(datasets) == 1)
        datasets = ["aime2024", "aime2025-i", "aime2025-ii", "gpqa"]
        
    model_semaphores = {model : asyncio.Semaphore(main_args.bucket_size) for model in models}
    
    for model in models:
        semaphore = model_semaphores[model]
        for dataset_name in datasets:
            model_name = MODEL_NAME_MAP[model]
            dataset = QuestionAnswerDataset(*DATASET_MAP[dataset_name][0])
            tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP_API[model])
            template_type = DATASET_MAP[dataset_name][1]
            for i, example in enumerate(dataset):
                specific_args = args[model][dataset_name]
                if main_args.decode_method == "bs":
                    for j in range(1, specific_args["max_num_beams"]):
                        file_path = f"../results/bs/{model_name}/{dataset_name}/{j+1}/{i}_0.json"
                        task_args = dict(specific_args)
                        task_args["num_beams"] = j+1
                        task = get_generation_and_store(task_args, file_path, 0, example, openai_client, tokenizer, template_type, semaphore, model, "bs")
                        tasks.append(task)
                elif main_args.decode_method == "vanilla":
                    for j in range(specific_args["max_num_samples"]):
                        file_path = f"../results/vanilla/{model_name}/{dataset_name}/{i}_{j}.json"
                        task = get_generation_and_store(specific_args, file_path, j, example, openai_client, tokenizer, template_type, semaphore, model, "vanilla")
                        tasks.append(task)
                elif main_args.decode_method == "vanilla_error_correction":
                    for j in range(specific_args["max_num_samples"]):
                        original_file_path = f"../results/vanilla/{model_name}/{dataset_name}/{i}_{j}.json"
                        # check if the trace at original_file_path was incorrect; only then proceed with correction
                        with open(original_file_path, "r") as f:
                            data = json.load(f)
                        if data["pred"] == data["answer"]:
                            continue    # predicted answer is correct
                        content = data["output"]
                        content = content[:int(len(content)*0.75)]
                        # clamp to last generated "\n"
                        if "\n" not in content:
                            continue
                        content = content[:content.rfind("\n")] + "\nWait, let me think again."
                        file_path = f"../results/vanilla_error_correction/{model_name}/{dataset_name}/{i}_{j}.json"
                        task = get_generation_and_store(specific_args, file_path, j, example, openai_client, tokenizer, template_type, semaphore, model, "vanilla", content)
                        tasks.append(task)
                else:
                    raise NotImplementedError
    await asyncio.gather(*tasks)
    
if __name__ == "__main__":
    asyncio.run(main())