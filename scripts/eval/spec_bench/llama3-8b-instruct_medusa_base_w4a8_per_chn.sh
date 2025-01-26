export CUDA_VISIBLE_DEVICES=3
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full-rotation
Model_id="llama-3-8b-instruct"

python3 evaluation/spec_bench/inference_medusa_base_w4a8_per_chn.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa_base_w4a8_per_chn_tree_32 \
    --memory-limit 0.8 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --medusa-num-heads 3 \
    --medusa-choices 'mc_sim_7b_31'
