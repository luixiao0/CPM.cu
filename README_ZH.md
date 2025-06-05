# CPM.cu

<strong>中文 | [English Version](./README.md)</strong>

CPM.cu 是一个针对端侧大模型推理设计的高效 CUDA 推理框架，核心支持 **稀疏架构**、**投机采样** 和 **低位宽量化** 等前沿技术创新。

<a href="https://github.com/OpenBMB/minicpm"><img src="https://img.shields.io/static/v1?label=MiniCPM4 项目&message=Web&color=green"></a> &ensp;
<a href="https://github.com/OpenBMB/cpm.cu/blob/main/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/cpm.cu">
</a>

<div id="news"></div>

## 🔥 项目进展

- [2025.06.06] 为 [MiniCPM4](https://github.com/openbmb/minicpm) 优化。
    - 支持 InfLLM-v2 注意力内核
    - 支持 MTP 层的滑动窗口，优化长上下文处理
    - 支持 MTP 层的量化
- [2025.05.29] 支持 [SpecMQuant](https://github.com/AI9Stars/SpecMQuant) 的量化。
    - 支持 LLM 的 Marlin GPTQ 内核
    - 支持量化 LLM 的推测采样
- [2025.03.01] 在 [FR-Spec](https://github.com/thunlp/FR-Spec) 发布首个版本。
    - 最先进的推测采样实现
    - 支持 FR-Spec：基于频率排序的推测采样
    - 支持 Flash-Attention 中的树形验证
    - 支持静态内存管理和内存重用
    - 支持融合内核
    - 支持分块预填充
    - 支持 CUDA Graph

<div id="demo"></div>

## 效果演示

![TODO 占位符](https://github.com/thunlp/Ouroboros/blob/main/figure/ouroboros.gif)

<div id="getstart"></div>

## 快速开始

- [安装](#install)
- [模型权重](#modelweights)
- [运行示例](#example)

<div id="install"></div>

## 安装

### 从源码安装

```bash
git clone https://github.com/OpenBMB/cpm.cu.git --recursive
cd cpm.cu
python3 setup.py install
```

<div id="modelweights"></div>

## 准备模型

请按照 [MiniCPM4](https://github.com/openbmb/minicpm) 的说明下载模型权重。

<div id="example"></div>

## 运行示例

我们提供了一个简单的示例来展示如何使用 CPM.cu。

```bash
python3 tests/test_generate.py
```

输出应为如下格式：

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

其中：

- `Prefill` (输入) 和 `Decode` (输出) 速度通过（长度、时间和 token/s）输出。
- `Mean accept length` (平均接受长度) 是使用投机采样时接受 token 的平均长度。
- `Decode token/s when acc = 1` 是将输出速度除以平均接受长度的结果。

## 致谢

我们的 `src/flash_attn` 文件夹基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v2.6.3/csrc/flash_attn) 并进行了修改。

我们从以下仓库中获取了实现灵感：

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

## 引用

如果您觉得我们的工作有价值，请引用我们的论文。

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