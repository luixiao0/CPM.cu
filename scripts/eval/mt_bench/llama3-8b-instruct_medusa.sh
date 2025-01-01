export CUDA_VISIBLE_DEVICES=3
Model_Path=/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full
Model_id="llama3-8b-instruct"

python3 evaluation/mt_bench/inference_medusa.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa_debug \
    --memory-limit 0.8 \
    --bench-name "mt_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
