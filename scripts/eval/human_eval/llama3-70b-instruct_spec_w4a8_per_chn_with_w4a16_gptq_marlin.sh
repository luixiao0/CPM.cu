export CUDA_VISIBLE_DEVICES=0
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-70B-Instruct-w4a8-gchn
Draft_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin
Model_id="llama-3-70b-instruct-w4a16-gptq_marlin-with_w4a16"

python3 evaluation/humaneval/inference_spec_w4a8_per_chn_with_w4a16_gptq_marlin.py \
    --model-path $Model_Path \
    --draft-path $Draft_Path \
    --cuda-graph \
    --model-id ${Model_id}_spec_iter_6 \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --spec-num-iter 6 \
    --draft-cuda-graph