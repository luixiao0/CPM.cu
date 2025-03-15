export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full
Model_id="llama-3-8b-instruct-w4a16-gptq_marlin"

python3 evaluation/spec_bench/inference_medusa_base_w4a16_gptq_marlin.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa \
    --memory-limit 0.80 \
    --bench-name "spec_bench" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --medusa-num-heads 3 \
    --medusa-choices 'mc_sim_7b_63_3head_top31'

