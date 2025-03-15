export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-70B-Instruct-rotation-gptq-pile/Meta-Llama-3-70B-Instruct-merge
Model_id="llama-3-70b-instruct-w4a8-qqq"

python3 evaluation/spec_bench/inference_baseline_w4a8_qqq.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --memory-limit 0.8 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
