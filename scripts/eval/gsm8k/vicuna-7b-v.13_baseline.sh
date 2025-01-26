export CUDA_VISIBLE_DEVICES=1
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
Model_Path=/home/ydzhang/checkpoints/lmsys/vicuna-7b-v1.3
Model_id="vicuna-7b-v1.3"

python3 evaluation/gsm8k/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}_baseline \
    --bench-path $Bench_Path \
    --memory-limit 0.8 \
    --dtype "float16"
