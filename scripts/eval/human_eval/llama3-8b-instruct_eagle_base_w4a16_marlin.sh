export CUDA_VISIBLE_DEVICES=0
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse_shrink0.4
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B
Model_id="llama-3-8b-instruct-w4a16-marlin"

python3 evaluation/humaneval/inference_eagle_w4a16_marlin.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_eagle_iter_3_tree_30 \
    --memory-limit 0.80 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 3 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 30