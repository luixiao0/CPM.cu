export CUDA_VISIBLE_DEVICES=3
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
Model_Path=/home/ydzhang/checkpoints/neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full
Model_id="llama3-8b-instruct"

python3 evaluation/gms8k/inference_medusa_base_w8a8.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa_base_w8a8 \
    --bench-path $Bench_Path \
    --memory-limit 0.8 \
    --dtype "float16" \
    --medusa-num-heads 3 \
    --medusa-choices 'mc_sim_7b_61'