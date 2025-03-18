export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-8B-Instruct-rotation-gptq-mse-pile-g128/Meta-Llama-3-8B-Instruct-merge
Model_id="llama-3-8b-instruct-w4a8-qqq_g128"

python3 evaluation/humaneval/inference_baseline_w4a8_qqq.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3"
