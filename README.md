# Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design

## Installation from source

```bash
conda create -n specmquant python==3.11 && conda activate specmquant
# install pytorch 
pip install -e .
```

## Evaluation

### Prepare

You can use an external quantization toolkit to quantize the model.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ): W4A16
- [QQQ](https://github.com/HandH1998/QQQ): W4A8-QQQ, W4A8-QQQ-g128
- [DeepCompressor](https://github.com/mit-han-lab/deepcompressor): W8A8, W4A8-QoQ, W4A8-QoQ-g128

For `W4A16`, `W4A8-QQQ`, `W4A8-QQQ-g128` and `W4A8-QoQ-g128`, after quantizing with the above toolkits you need to convert the model checkpoints using the scripts in `scripts/model_convert`.

For the models applied with rotation method, you need to convert the eagle checkpoint using the scripts `scripts/model_convert/convert_eagle_rotation.sh`.

### Run Evaluation

#### MT bench

All scripts for evaluation are located in the `scripts/eval/mt_bench` folder. Here we use Llama-3-8B-Instruct as an example:

```bash
# 1. Run evaluations
bash scripts/eval/mt_bench/llama3-8b-instruct/<precision>/run_baseline.sh
bash scripts/eval/mt_bench/llama3-8b-instruct/<precision>/run_eagle.sh

# 2. Evaluate speed
bash scripts/mt_bench/llama3-8b-instruct/speed_up.sh

```

Replace `<precision>` with one of: `fp16`, `w4a16`, `w4a8-qqq`, `w4a8-qqq-g128`, `w4a8-qoq`, or `w4a8-qoq-g128`.

#### Spec bench

Spec bench evaluation for W4A16 Llama-3-70B-Instruct

```bash
# 1. Run evaluations
bash scripts/eval/spec_bench/llama3-70b-instruct-w4a16/run_baseline.sh
bash scripts/eval/spec_bench/llama3-70b-instruct-w4a16/run_spec.sh
bash scripts/eval/spec_bench/llama3-70b-instruct-w4a16/run_eagle.sh
bash scripts/eval/spec_bench/llama3-70b-instruct-w4a16/run_hierspec.sh


# 2. Evaluate speed
bash scripts/eval/spec_bench/llama3-70b-instruct-w4a16/speedup.sh

```

#### Performance evaluation

We provide the performance evaluation for `gsm8k` and `human_eval`.

```bash
# 1. Run evaluations
bash scripts/eval/<benchmark>/llama3-8b-instruct/<precision>/run_baseline.sh

# 2. Evaluate preformance
bash scripts/eval/<benchmark>/llama3-8b-instruct/check_correctness.sh
```

