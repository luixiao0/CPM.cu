export CUDA_VISIBLE_DEVICES=3
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B-w4a8_rotation
Model_id="llama-3-8b-instruct-w4a8_per_chn"

python3 evaluation/gsm8k/inference_eagle_w4a8_per_chn.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_eagle_iter_6_tree_60 \
    --bench-path $Bench_Path \
    --memory-limit 0.80 \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60