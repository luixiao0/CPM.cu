export CUDA_VISIBLE_DEVICES=3
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-qqq-rotation-4bit-128g-pileval-mse_merge
Draft_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-rotation-4bit-128g-pileval-mse-desc-static_group_default_merge
Model_id="llama-3-70b-instruct-w4a16-gptq_marlin"

python3 evaluation/spec_bench/inference_spec_w4a16_gm_for_w4a16_gm.py \
    --model-path $Model_Path \
    --draft-path $Draft_Path \
    --cuda-graph \
    --model-id ${Model_id}_spec_w4a16_gm_iter_6 \
    --memory-limit 0.8 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --spec-num-iter 6 \
    --draft-cuda-graph