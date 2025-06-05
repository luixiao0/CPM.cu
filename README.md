# CPM.cu

<strong>[中文](./README_ZH.md) |
English</strong>

CPM.cu is a high-performance CUDA implementation for LLMs, optimized for MiniCPM4 and featuring cutting-edge techniques in sparse architecture, speculative sampling and quantization.

<a href="https://github.com/OpenBMB/minicpm"><img src="https://img.shields.io/static/v1?label=MiniCPM4 Project&message=Web&color=green"></a> &ensp;
<a href="https://github.com/OpenBMB/cpm.cu/blob/main/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/cpm.cu">
</a>

<div id="news"></div>

## 🔥 Project Updates

- [2025.06.06] Optimized for [MiniCPM4](https://github.com/openbmb/minicpm).
    - Support InfLLM-v2 attention kernel
    - Support sliding-window for the MTP layer, optimized for long context
    - Support quantization for the MTP layer
- [2025.05.29] Support Quantization at [SpecMQuant](https://github.com/AI9Stars/SpecMQuant).
    - Support Marlin GPTQ kernel for the LLM
    - Support Speculative Sampling for quantized LLM
- [2025.03.01] Release the first version at [FR-Spec](https://github.com/thunlp/FR-Spec).
    - SOTA Speculative Sampling Implementation
    - Support FR-Spec: Frequency-Ranked Speculative Sampling
    - Support Tree-based verification in Flash-Attention
    - Support Static memory management and memory reuse
    - Support Fused kernels
    - Support Chunked prefill
    - Support CUDA Graph

<div id="demo"></div>

## Demo

![TODO placeholder](https://github.com/thunlp/Ouroboros/blob/main/figure/ouroboros.gif)

<div id="getstart"></div>

## Getting Started

- [Installation](#install)
- [Model Weights](#modelweights)
- [Quick Start](#example)

<div id="install"></div>

## Installation

### Install from source

```bash
git clone https://github.com/OpenBMB/cpm.cu.git --recursive
cd cpm.cu
python3 setup.py install
```

<div id="modelweights"></div>

## Prepare Model

Please follow [MiniCPM4](https://github.com/openbmb/minicpm) to download the model weights.

<div id="example"></div>

## Quick Start

We provide a simple example to show how to use CPM.cu to generate text.

```bash
python3 tests/test_generate.py
```

The output should be of the following format:

```bash
Generated text (streaming output):
--------------------------------------------------
Prefilling: 100.0% (14774/14774 tokens) @ 6675.6 tokens/s - Complete!

<Generated Output HERE>
==================================================
Stream Generation Summary:
==================================================
Prefill length: 14774
Prefill time: 2.35 s
Prefill tokens/s: 6291.27
Decode length: 11
Decode time: 0.07 s
Decode tokens/s: 150.57
Mean accept length: 2.17
Decode token/s when acc = 1: 69.49
```

Where the Prefill and Decode speed are output, and the Mean accept length is the average length of the accepted tokens when using Speculative Sampling.
The Decode token/s when acc = 1 is the result of dividing the Decode speed by the Mean accept length.

## Acknowledgments

We have drawn inspiration from the following repositories for some code and implementation ideas:

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)

## Citation

Please cite our paper if you find our work valuable.

```
@article{zhao2025fr,
  title={FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling},
  author={Zhao, Weilin and Pan, Tengyu and Han, Xu and Zhang, Yudi and Sun, Ao and Huang, Yuxiang and Zhang, Kaihuo and Zhao, Weilun and Li, Yuxuan and Wang, Jianyong and others},
  journal={arXiv preprint arXiv:2502.14856},
  year={2025}
}

@article{zhang2025specmqaunt,
  title={Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design},
  author={Zhang, Yudi and Zhao, Weilin and Han, Xu and Zhao, Tiejun and Xu, Wang and Cao, Hailong and Zhu, Conghui},
  journal={arXiv preprint arXiv:2505.22179},
  year={2025}
}

@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM},
  year={2025}
}
```