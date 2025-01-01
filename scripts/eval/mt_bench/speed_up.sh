spec_file="data/mt_bench/model_answer/llama3-8b-instruct_medusa_base_w8a8.jsonl"
base_file="data/mt_bench/model_answer/llama3-8b-instruct_w8a8_baseline.jsonl"
tokenizer_path="/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct"
python evaluation/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path