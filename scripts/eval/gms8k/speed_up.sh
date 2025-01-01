spec_file="data/gms8k/model_answer/llama3-8b-instruct_medusa_base_w8a8.jsonl"
base_file="data/gms8k/model_answer/llama3-8b-instruct_baseline_w8a8.jsonl"
tokenizer_path="/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct"
python evaluation/gms8k/speed_gms8k.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path