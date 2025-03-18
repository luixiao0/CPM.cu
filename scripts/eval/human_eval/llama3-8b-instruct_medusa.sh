export CUDA_VISIBLE_DEVICES=1
Model_Path=/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
Meudsa_Path=/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full
Model_id="llama-3-8b-instruct"

python3 evaluation/humaneval/inference_medusa.py \
    --model-path $Model_Path \
    --medusa-path $Meudsa_Path \
    --cuda-graph \
    --model-id ${Model_id}_medusa \
    --memory-limit 0.80 \
    --bench-name "human_eval_evalplus" \
    --dtype "float16" \
    --chat-template "llama-3" \
    --medusa-num-heads 3 \
    --medusa-choices 'mc_sim_7b_61'

