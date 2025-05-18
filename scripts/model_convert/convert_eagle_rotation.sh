EAGLE_Path=models/EAGLE-LLaMA3-Instruct-8B
Base_Model_Path=meta-llama/Meta-Llama-3-8B-Instruct
Rotatin_Path=models/Meta-Llama-3-8B-Instruct-w4a8-qqq/rotation.pth
Output_Path=models/EAGLE-LLaMA3-Instruct-8B-on-w4a8-qqq

python model_convert/convert_eagle_rotation.py \
    --eagle-path $EAGLE_Path \
    --base-model-path $Base_Model_Path \
    --rotation-path $Rotatin_Path \
    --output-path $Output_Path