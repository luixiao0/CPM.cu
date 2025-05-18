Quant_Path=quant_models/Meta-Llama-3-8B-Instruct-w4a8-qoq
Output_Path=models/Meta-Llama-3-8B-Instruct-w4a8-qoq

python model_convert/convert_w4a8_qqq.py \
    --quant-path $Quant_Path \
    --output-path $Output_Path 