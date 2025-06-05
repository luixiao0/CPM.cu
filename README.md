<div align="center">

<h1>CPM.cu</h1>

**A Efficient CUDA-based inference framework for CPM models**

  <strong>[中文](./README_ZH.md) |
  English</strong>

  <p align="center">
    <a href="#news">News</a> • <a href="#demo">Demo</a> • <a href="#install">Installation</a> • <a href="#example">Quick Start</a>
    <br>
  </p>

  <a href="https://github.com/OpenBMB/cpm.cu/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/cpm.cu">
  </a>
  <img alt="Github" src="https://img.shields.io/github/downloads/OpenBMB/cpm.cu/total">

</div>

<div id="news"></div>

## News

- [2025.06.06] Optimized for MiniCPM4.
    - Implemented InfLLM-v2 Attention Kernel
    - Long Context Optimization for Speculative Sampling
- [2025.05.29] Support Quantization at https://github.com/AI9Stars/SpecMQuant.
    - Quantization support
    - Combine Quantization and Speculative Sampling
- [2025.03.01] Release the first version of the CUDA inference framework at https://github.com/thunlp/FR-Spec.
    - SOTA Speculative Sampling Speed
    - Tree-based verification implemented in Flash-Attention for Speculative Sampling
    - Static memory management and memory reuse
    - Fused kernels
    - Chunked prefill
    - CUDA Graph support

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

## Acknowledgments

We have drawn inspiration from the following repositories for some code and implementation ideas:

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)

## Citation

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