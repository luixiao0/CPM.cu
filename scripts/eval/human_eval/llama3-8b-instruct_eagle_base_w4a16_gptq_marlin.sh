export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-rotation-4bit-128g-pileval-mse-desc-static_group_default_merge
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B-w4a8_rotation
Model_id="llama-3-8b-instruct-w4a16-gptq_marlin"

python3 evaluation/humaneval/inference_eagle_w4a16_gptq_marlin_rot.py \
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