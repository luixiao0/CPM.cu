export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse_shrink0.4
Model_id="llama-3-8b-instruct-w4a16"

python3 evaluation/spec_bench/inference_baseline_w4a16.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --memory-limit 0.8 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
