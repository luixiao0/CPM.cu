export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-70B-Instruct-rotation-gptq-pile-g128/Meta-Llama-3-70B-Instruct-merge
Draft_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-8B-Instruct-rotation-gptq-mse-pile/Meta-Llama-3-8B-Instruct-merge
Model_id="llama-3-70b-instruct-w4a8-qqq_g128"

python3 evaluation/humaneval/inference_spec_w4a8_qqq_for_w4a8_qqq.py \
    --model-path $Model_Path \
    --draft-path $Draft_Path \
    --cuda-graph \
    --model-id ${Model_id}_spec_w4a8_qqq_iter_6 \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --spec-num-iter 6 \
    --draft-cuda-graph