export CUDA_VISIBLE_DEVICES=3
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-gptq_marlin
Draft_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin
Model_id="llama-3-70b-instruct-w4a16-gptq_marlin"

python3 evaluation/spec_bench/inference_spec_w4a16_gptq_marlin.py \
    --model-path $Model_Path \
    --draft-path $Draft_Path \
    --cuda-graph \
    --model-id ${Model_id}_spec_iter_5 \
    --memory-limit 0.8 \
    --bench-name "mt_bench" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --spec-num-iter 5 \
    --draft-cuda-graph