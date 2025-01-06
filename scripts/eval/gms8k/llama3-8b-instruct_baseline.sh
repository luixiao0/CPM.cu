export CUDA_VISIBLE_DEVICES=0
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
Model_Path=/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
Model_id="llama-3-8b-instruct"

python3 evaluation/gms8k/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --bench-path $Bench_Path \
    --memory-limit 0.8 \
    --dtype "float16"
