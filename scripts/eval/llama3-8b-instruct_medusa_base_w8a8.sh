export CUDA_VISIBLE_DEVICES=2
Model_Path=/home/ydzhang/checkpoints/neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full
Model_id="llama3-8b-instruct"

python3 evaluation/inference_medusa_base_w8a8.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa_base_w8a8 \
    --memory-limit 0.8 \
    --bench-name "mt_bench" \
    --dtype "float16" \
    --chat-template "llama-3"
