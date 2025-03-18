export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B
Model_id="llama-3-8b-instruct"

python3 evaluation/humaneval/inference_eagle.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_eagle_iter_6_tree_60 \
    --memory-limit 0.80 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60