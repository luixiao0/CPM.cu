export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w8a8-gchn-pileval-neuralmagic
Model_id="llama-3-8b-instruct"

python3 evaluation/humaneval/inference_baseline_w8a8.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_w8a8_baseline \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3"
