export CUDA_VISIBLE_DEVICES=1
Bench_Path=/home/ydzhang/data/openai/gsm8k/socratic/
Model_Path=/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full
Model_id="llama-3-8b-instruct"

python3 evaluation/gsm8k/inference_medusa_base_w4a8_per_chn.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa_base_w4a8_per_chn_pileval_tree_64 \
    --bench-path $Bench_Path \
    --memory-limit 0.8 \
    --dtype "float16" \
    --medusa-num-heads 3 \
    --medusa-choices 'mc_sim_7b_all1'