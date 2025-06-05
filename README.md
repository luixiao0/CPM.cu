# CPM.cu

<div align="center">

**A CUDA-based inference framework for CPM models**

  <strong>[中文](./README_ZH.md) |
  English</strong>

</div>

## News

- [2025.06.06] Support MiniCPM4.
- [2025.05.29] Support Quantization at https://github.com/AI9Stars/SpecMQuant.
- [2025.03.01] First version of the CUDA inference framework at https://github.com/thunlp/FR-Spec.

## Install

### Installation from source

```bash
git clone https://github.com/OpenBMB/cpm.cu.git --recursive
cd cpm.cu
python3 setup.py install
```

## Example

```bash
python3 tests/test_generate.py
```