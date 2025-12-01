import pandas as pd
import re
import os
import pickle as pkl

def cache_output(return_output, file_name, prefix="."):
    os.makedirs(os.path.join(prefix,".cache"), exist_ok=True)
    with open(os.path.join(prefix, f".cache/{file_name}"), "wb") as f:
        pkl.dump(return_output, f)
        
def read_cache(file_name, prefix="."):
    with open(os.path.join(prefix, f".cache/{file_name}"), "rb") as f:
        return_output = pkl.load(f)
    return return_output

def regex_rfind(pattern, text):
    match = re.findall(pattern, text)
    if match:
        return match[-1]
    return None

def read_json(json_file_path):
    data = pd.read_json(json_file_path, lines=True)
    out = {}
    for col in data.columns:
        out[col] = data.loc[0, col]
    return out

def isint(x):
    if isinstance(x, str) and "_" in x:
        return False
    try:
        int(x)
        return True
    except:
        return False

def process_problem(problem):
    if problem.startswith("What is the correct answer to this question: "):
        problem = problem[len("What is the correct answer to this question: "):]
    if problem.endswith("\nAnswer: (A), (B), (C) or (D) choose the correct option within\\boxed{}"):
        problem = problem[:-len("\nAnswer: (A), (B), (C) or (D) choose the correct option within\\boxed{}")]
    if problem.endswith(" Please reason step by step, and put your final answer within \\boxed{}."):
        problem = problem[:-len(" Please reason step by step, and put your final answer within \\boxed{}.")]
    if problem.endswith("Answer: (A), (B), (C) or (D) choose within\\boxed{}"):
        problem = problem[:-len("Answer: (A), (B), (C) or (D) choose within\\boxed{}")]
    return problem

def break_into_question_and_options(problem):
    locations = (
        problem.rfind("\n(A) "), 
        problem.rfind("\n(B) "), 
        problem.rfind("\n(C) "), 
        problem.rfind("\n(D) "),
    )
    choices = (
        problem[locations[0]+5:locations[1]+1].strip(),
        problem[locations[1]+5:locations[2]+1].strip(),
        problem[locations[2]+5:locations[3]+1].strip(),
        problem[locations[3]+5:].strip(),
    )
    return problem.split("\nChoices:")[0], choices

def add_template(problem, template_type):
    if template_type == "mcq":
        processed = "What is the correct answer to this question: "+problem+"\nAnswer: (A), (B), (C), or (D). Choose the correct option within \\boxed{}.\n"
    elif template_type == "subj":
        processed = problem+" Please reason step by step and put your answer within \\boxed{}.\n"
    else:
        raise NotImplementedError
    return processed

def get_first_nonnull(match):
    if isinstance(match, tuple):
        for x in match:
            if x:
                return x
    return match

def parse_output(output, template_type):
    if template_type == "subj":
        pattern = r"\\boxed{(\d\d\d|\d\d|\d)}"
    elif template_type == "mcq":
        pattern = r"\\boxed{\s*(A|B|C|D)|" \
                    r"\\boxed{\s*\((A|B|C|D)\)|" \
                    r"\\boxed{\s*\\text{\s*\((A|B|C|D)\)|" \
                    r"\\boxed{\s*\\text{\s*(A|B|C|D)|" \
                    r"\*\*(A|B|C|D)|" \
                    r"\*\*\((A|B|C|D)"
    else:
        raise NotImplementedError
    match = regex_rfind(pattern, output)
    match = get_first_nonnull(match)
    if isint(match):
        # To convert integer-like strings to ints so that comparison is integer equality
        match = int(match)
    return match or "NO_ANSWER"
