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

def get_function_name(question: str, lang: str = 'Python'):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix


def extract_generation_code(question, output, verbose: bool=False):
    setting = {
        'full_name': 'Python',
        'indent': 4,
    }
    lang = setting['full_name']
    indent = setting['indent']

    try:
        code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        
        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
        
        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent-1] == ' ':
                indent += 1
            
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]
    
        generation = func_prefix + '\n' + body + '\n'
        from IPython import embed; embed()
        result = generation

    except Exception as ex:
        result = question + '\n' + output
    
    return result


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


def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def gen_prompt(prompt: str) -> str:
#     if args.model_type == "deepseek":
#         return '''
# Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
# ```{}
# {}
# ```
# '''.strip().format("Python", prompt.strip())
    prompt = (
        "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
        + prompt
    )
    return prompt


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


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"



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
    #     prompt = gen_prompt(prompt)
    #     # if 'deepseek' in model_id:
    #     #     input_str = deepseek_temp.format(prompt=prompt[:113], prefix=prompt[113:])
    #     #     prompt = input_str
    #     # elif 'codellama' in model_id:
    #     #     prompt = "[INST] " + prompt[:113] + "[/INST]\n" + prompt[113:]
        
    #     conv = get_conversation_template(converse_template)
    #     if "llama-2" in converse_template or "llama-3" in converse_template:
    #         sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    #         conv.system_message = sys_p
    #     conv.append_message(conv.roles[0], prompt[:113])
    #     conv.append_message(conv.roles[1], prompt[113:])
    #     prompt = conv.get_prompt()

    #     input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda").view(1, -1)
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
            prompt = gen_prompt(prompt)
            # completion = model.run(prompt)

            # if 'deepseek' in model_id:
            #     input_str = deepseek_temp.format(prompt=prompt[:113], prefix=prompt[113:])
            #     prompt = input_str
            # elif 'codellama' in model_id:
            #     prompt = "[INST] " + prompt[:113] + "[/INST]\n" + prompt[113:]

            conv = get_conversation_template(converse_template)
            # if "llama-2" in converse_template or "llama-3" in converse_template:
            if "llama-2" in converse_template:
                sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                conv.system_message = sys_p
            conv.messages = []
            conv.append_message(conv.roles[0], prompt[:113])
            conv.append_message(conv.roles[1], prompt[113:]+_MAGIC_SPLITTER_)
            prompt = conv.get_prompt()
            prompt = prompt.split(_MAGIC_SPLITTER_)[0]
            
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda").view(1, -1)
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, step, accept_length_tree = forward_func(
                input_ids, 
                model, 
                tokenizer, 
                max_new_tokens, 
                max_length
            )
            torch.cuda.synchronize()
            cur_time = time.time() - start_time
            accept_lengths_tree.extend(accept_length_tree)
            
            completion = tokenizer.decode(output_ids, skip_special_tokens=True)
            completion = fix_indents(completion)

            eval_sample = dict(task_id=task_id, completion=filter_code(completion)) 
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
                    "wall_time": cur_time,
                    "accept_lengths": cur_accept_lengths_tree,
                    "generate_speed": int(new_token) / cur_time,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
            
            total_new_tokens += new_token
            total_time += cur_time
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
