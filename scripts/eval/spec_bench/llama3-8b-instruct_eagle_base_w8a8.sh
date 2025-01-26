export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B
Model_id="llama-3-8b-instruct-w8a8"

python3 evaluation/spec_bench/inference_eagle_w8a8.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_eagle_iter_6_tree_60 \
    --memory-limit 0.80 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60