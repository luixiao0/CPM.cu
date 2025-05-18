Model_Path=/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-8B-Instruct-rotation-gptq-mse-pile-g128/Meta-Llama-3-8B-Instruct-merge
Eagle_Path=/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8-w4a8_qqq_gchn_rotation
Model_id="llama-3-8b-instruct"
Bench_name="gsm8k"

python3 evaluation/inference_eagle_w4a8_qqq.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qqq-g128/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60 \
    --max-new-tokens 256