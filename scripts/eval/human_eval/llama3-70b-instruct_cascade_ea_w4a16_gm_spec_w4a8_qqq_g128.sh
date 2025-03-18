export CUDA_VISIBLE_DEVICES=0
Model_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-70B-Instruct-rotation-gptq-pile-g128/Meta-Llama-3-70B-Instruct-merge
Draft_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-rotation-4bit-128g-pileval-mse-desc-static_group_default_merge
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B-w4a8_rotation
Model_id="llama-3-70b-instruct-w4a8-qqq_g128"

python3 evaluation/humaneval/inference_csc_eagle_w4a16_gm_rot_spec_w4a8_qqq.py \
    --model-path $Model_Path \
    --draft-path $Draft_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_cascade_eagle_iter_3_w4a16_gm_spec_min_draft_6 \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --spec-min-draft-length 6 \
    --draft-cuda-graph \
    --eagle-num-iter 3 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 30