export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-qqq-rotation-4bit-128g-pileval-mse_merge
Draft_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-8B-Instruct-rotation-gptq-mse-pile/Meta-Llama-3-8B-Instruct-merge
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8-w4a8_qqq_gchn_rotation
Model_id="llama-3-70b-instruct-w4a16-gptq_marlin"

python3 evaluation/humaneval/inference_csc_eagle_w4a8_qqq_rot_spec_w4a16_gm.py \
    --model-path $Model_Path \
    --draft-path $Draft_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}_cascade_eagle_iter_3_w4a8_qqq_spec_min_draft_6 \
    --memory-limit 0.8 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --spec-min-draft-length 6 \
    --draft-cuda-graph \
    --eagle-num-iter 3 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 30