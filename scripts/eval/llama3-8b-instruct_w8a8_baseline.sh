export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8
Model_id="llama3-8b-instruct"

python3 evaluation/inference_w8a8_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_w8a8_baseline \
    --memory-limit 0.8 \
    --bench-name "mt_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
