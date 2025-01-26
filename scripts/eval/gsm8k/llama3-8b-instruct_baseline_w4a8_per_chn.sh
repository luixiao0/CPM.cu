export CUDA_VISIBLE_DEVICES=3
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
# Model_Path=/home/ydzhang/checkpoints/neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval
Model_id="llama-3-8b-instruct"

python3 evaluation/gsm8k/inference_baseline_w4a8_per_chn.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline_w4a8_gchn_pileval \
    --bench-path $Bench_Path \
    --memory-limit 0.8 \
    --dtype "float16"
