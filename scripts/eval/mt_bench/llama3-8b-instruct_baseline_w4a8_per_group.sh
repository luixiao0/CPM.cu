export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-g128-merge
Model_id="llama-3-8b-instruct"

python3 evaluation/spec_bench/inference_baseline_w4a8_per_group.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_w4a8_per_group_baseline \
    --memory-limit 0.8 \
    --bench-name "mt_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
