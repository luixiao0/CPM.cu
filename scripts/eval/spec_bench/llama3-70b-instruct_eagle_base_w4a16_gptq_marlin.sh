export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-qqq-rotation-4bit-128g-pileval-mse_merge
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-70B-qqq_rotation
Model_id="llama-3-70b-instruct-w4a16-gptq_marlin"

python3 evaluation/spec_bench/inference_eagle_w4a16_gptq_marlin_rot.py \
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