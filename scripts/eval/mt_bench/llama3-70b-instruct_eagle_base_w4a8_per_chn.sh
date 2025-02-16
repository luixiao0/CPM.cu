export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-70B-Instruct-w4a8-gchn
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-70B-w4a8_rotation
Model_id="llama-3-70b-instruct-w4a8_per_chn"

python3 evaluation/spec_bench/inference_eagle_w4a8_per_chn.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_eagle_iter_3_tree_30 \
    --memory-limit 0.80 \
    --bench-name "mt_bench" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 3 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 30