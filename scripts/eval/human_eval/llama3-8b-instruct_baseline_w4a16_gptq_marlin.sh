export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-rotation-4bit-128g-pileval-mse-desc-static_group_default_merge
Model_id="llama-3-8b-instruct-w4a16-gptq_marlin"

python3 evaluation/humaneval/inference_baseline_w4a16_gptq_marlin.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3"
