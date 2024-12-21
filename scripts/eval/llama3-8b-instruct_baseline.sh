export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
Model_id="llama3-8b-instruct"

python3 evaluation/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --memory-limit 0.8 \
    --bench-name "mt_bench" \
    --dtype "bfloat16" \
    --chat-template "llama-3"
