import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path


def speed(jsonl_file, jsonl_file_base, checkpoint_path, task="gms8k", report=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    forward_time_list = []
    correct_list = []
    for datapoint in data:
        tokens = datapoint["new_tokens"]
        times = datapoint["wall_time"]
        accept_lengths_list.extend(datapoint["accept_lengths"])
        speeds.append(tokens/times)
        forward_times = datapoint["steps"]
        forward_time_list.append(times/forward_times)
        correct_list.append(datapoint["correct"])


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    base_correct_list = []
    for datapoint in data:
        tokens = datapoint["new_tokens"]
        times = datapoint["wall_time"]
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens
        base_correct_list.append(datapoint["correct"])  

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()

    accuracy = sum(correct_list) / len(correct_list)
    accuracy_base = sum(base_correct_list) / len(base_correct_list)

    if report:
        print("="*30, "Task: ", task, "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
        print("Accuracy: ", accuracy)
        print("Accuracy for the baseline: ", accuracy_base)
        print("Avg time per decode step: ", np.mean(forward_time_list))
        
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        default='../data/mini_bench/model_answer/vicuna-7b-v1.3-eagle-float32-temperature-0.0.jsonl',
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default='../data/mini_bench/model_answer/vicuna-7b-v1.3-vanilla-float32-temp-0.0.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="The file path of evaluated baseline.",
    )

    args = parser.parse_args()
    speed(jsonl_file=args.file_path, jsonl_file_base=args.base_path, checkpoint_path=args.checkpoint_path)