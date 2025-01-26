spec_file="data/gsm8k/model_answer/llama3-8b-instruct_medusa_base_w8a8.jsonl"
base_file="data/gsm8k/model_answer/llama3-8b-instruct_baseline_w8a8.jsonl"
tokenizer_path="/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct"
python evaluation/gsm8k/speed_gsm8k.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path