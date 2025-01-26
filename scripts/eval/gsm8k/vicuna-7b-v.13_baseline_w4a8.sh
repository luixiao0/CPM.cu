export CUDA_VISIBLE_DEVICES=3
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
Model_Path=/home/ydzhang/checkpoints/deepcompress/vicuna-7b-v1.3-w4a8-gchn-pileval
Model_id="vicuna-7b-v1.3"

python3 evaluation/gsm8k/inference_baseline_w4a8_per_chn.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline_w4a8 \
    --bench-path $Bench_Path \
    --memory-limit 0.8 \
    --dtype "float16"
