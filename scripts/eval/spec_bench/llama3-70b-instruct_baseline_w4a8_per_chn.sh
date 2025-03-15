export CUDA_VISIBLE_DEVICES=3
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-70B-Instruct-w4a8-gchn
Model_id="llama-3-70b-instruct-w4a8-per_chn"

python3 evaluation/spec_bench/inference_baseline_w4a8_per_chn.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --memory-limit 0.8 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
