Model_Path=models/Meta-Llama-3-8B-Instruct-w4a8-qqq
Eagle_Path=models/EAGLE-LLaMA3-Instruct-8B-on-w4a8-qqq
Model_id="llama-3-8b-instruct"
Bench_name="mt_bench"

python3 evaluation/inference_eagle_w4a8_qqq.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qqq/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60