import os
import json
from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from transformers import AutoTokenizer

import time, torch

import argparse, re
import numpy as np

from fastchat.model import get_conversation_template
from evalplus.sanitize import sanitize


def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )

    return results



def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


def test_fix_indents():
    text = "   # TODO: Implement separate_paren_groups\nreturn []"
    print(fix_indents(text))


######

# borrow from evalplus

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]
EOS += ["\n```\n"]


# Model instructions
instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]

    return task_prompt


@torch.inference_mode()
def run_eval(
    model,
    tokenizer,
    data_path,
    forward_func,
    model_id,
    answer_file,
    max_new_tokens,
    max_length,
    teminators,
    **kwargs,
):
    converse_template = kwargs.pop('chat_template', 'llama-3')

    dataset = read_problems(data_path)
    # n_sample = kwargs.get("n_sample", 1) # TODO: n_samples in kwargs
    n_sample = 1
    # best_temperature = {1: 0.1, 10: 0.6, 100: 0.8}

    # entry = dataset['HumanEval/0']
    # warmup_times = 3
    # for wm_i in range(warmup_times):
    #     prompt = entry["prompt"]
    #     # prompt = gen_prompt(prompt)
    #     # if 'deepseek' in model_id:
    #     #     input_str = deepseek_temp.format(prompt=prompt[:113], prefix=prompt[113:])
    #     #     prompt = input_str
    #     # elif 'codellama' in model_id:
    #     #     prompt = "[INST] " + prompt[:113] + "[/INST]\n" + prompt[113:]
        
    #     conv = get_conversation_template(converse_template)
    #     if "llama-2" in converse_template or "llama-3" in converse_template:
    #         sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    #         conv.system_message = sys_p
    #     conv.messages = []
        
    #     task_prompt = make_raw_chat_prompt(
    #         prompt,
    #         instruction_prefix,
    #         response_prefix,
    #         tokenizer,
    #         conv,
    #     )

    #     input_ids = tokenizer.encode(task_prompt, return_tensors='pt').to("cuda").view(1, -1)
    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #     output_ids, new_token, step, accept_length_tree = forward_func(
    #         input_ids, 
    #         model, 
    #         tokenizer, 
    #         max_new_tokens, 
    #         max_length
    #     )
    #     torch.cuda.synchronize()
    #     cur_time = time.time() - start_time
    #     print(f"warmup {wm_i} done")


    eval_samples = []
    accept_lengths_tree = []
    total_new_tokens = 0
    total_time = 0
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    for task_id in dataset:
        for smaple_id in range(n_sample):
            cur_accept_lengths_tree = []
            prompt = dataset[task_id]["prompt"]
            # prompt = gen_prompt(prompt)
            # completion = model.run(prompt)

            # if 'deepseek' in model_id:
            #     input_str = deepseek_temp.format(prompt=prompt[:113], prefix=prompt[113:])
            #     prompt = input_str
            # elif 'codellama' in model_id:
            #     prompt = "[INST] " + prompt[:113] + "[/INST]\n" + prompt[113:]

            
            task_prompt = make_raw_chat_prompt(
                prompt,
                instruction_prefix,
                response_prefix,
                tokenizer,
            )
            
            input_ids = tokenizer.encode(task_prompt, return_tensors='pt').to("cuda").view(1, -1)
            # torch.cuda.synchronize()
            # start_time = time.time()
            output_ids, new_token, step, accept_length_tree, decode_time = forward_func(
                input_ids, 
                model, 
                tokenizer, 
                max_new_tokens, 
                max_length,
                teminators
            )
            # torch.cuda.synchronize()
            # cur_time = time.time() - start_time
            accept_lengths_tree.extend(accept_length_tree)
            
            completion = tokenizer.decode(output_ids, skip_special_tokens=True)
            min_index = 10000
            for eos in EOS:
                if eos in completion:
                    min_index = min(min_index, completion.index(eos))
            completion = completion[:min_index].replace("\t", "    ")
            # fix_completion = fix_indents(completion)
            sanitized_completion = sanitize(completion, dataset[task_id]["entry_point"])

            eval_sample = dict(task_id=task_id, completion=sanitized_completion) 
            eval_samples.append(eval_sample)

            cur_accept_lengths_tree.extend(accept_length_tree)
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "data_id": task_id,
                    "model_id": model_id,
                    "model_output": completion,
                    "steps": step,
                    "new_tokens": int(new_token),
                    "wall_time": decode_time,
                    "accept_lengths": cur_accept_lengths_tree,
                    "generate_speed": int(new_token) / decode_time,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
            
            total_new_tokens += new_token
            total_time += decode_time
            progress_bar.update(1)

    result = None
    pred_dir =  os.path.dirname(answer_file)
    pred_filename = f"{pred_dir}/humaneval_predictions.jsonl"
    write_jsonl(pred_filename, eval_samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
    print(result)
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    print("#Generate latency: ", total_new_tokens / total_time)



deepseek_temp = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{prompt}\n### Response:\n{prefix}"
