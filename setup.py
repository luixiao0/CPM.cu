import os, glob
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

def detect_cuda_arch():
    """自动检测当前CUDA架构"""
    # 1. 首先检查环境变量是否指定了架构
    env_arch = os.getenv("LLAMACU_CUDA_ARCH")
    if env_arch:
        # 仅支持逗号分隔的简单格式，如 "80,86"
        arch_list = []
        tokens = env_arch.split(',')
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
                
            # 检查格式：必须是纯数字
            if not token.isdigit():
                raise ValueError(
                    f"Invalid CUDA architecture format: '{token}'. "
                    f"LLAMACU_CUDA_ARCH should only contain comma-separated numbers like '80,86'"
                )
            
            arch_list.append(token)
        
        if arch_list:
            print(f"Using CUDA architectures from environment variable: {arch_list}")
            return arch_list
    
    # 2. 检查是否有torch库，如果有则自动检测
    try:
        import torch
    except ImportError:
        # 3. 如果没有环境变量也没有torch，报错
        raise RuntimeError(
            "CUDA architecture detection failed. Please either:\n"
            "1. Set environment variable LLAMACU_CUDA_ARCH (e.g., export LLAMACU_CUDA_ARCH=90), or\n"
            "2. Install PyTorch (pip install torch) for automatic detection.\n"
            "Common CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)"
        )
    
    # 使用torch自动检测所有GPU架构
    try:
        if torch.cuda.is_available():
            arch_set = set()
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                major, minor = torch.cuda.get_device_capability(i)
                arch = f"{major}{minor}"
                arch_set.add(arch)
            
            arch_list = sorted(list(arch_set))  # 排序保证一致性
            print(f"Detected CUDA architectures: {arch_list} (from {device_count} GPU devices)")
            return arch_list
        else:
            raise RuntimeError(
                "No CUDA devices detected. Please either:\n"
                "1. Set environment variable LLAMACU_CUDA_ARCH (e.g., export LLAMACU_CUDA_ARCH=90), or\n"
                "2. Ensure CUDA devices are available and properly configured.\n"
                "Common CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)"
            )
    except Exception as e:
        raise RuntimeError(
            f"CUDA architecture detection failed: {e}\n"
            "Please set environment variable LLAMACU_CUDA_ARCH (e.g., export LLAMACU_CUDA_ARCH=90).\n"
            "Common CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)"
        )

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "16"
    return nvcc_extra_args + ["--threads", nvcc_threads]

def get_compile_args():
    """根据是否为debug模式返回不同的编译参数"""
    debug_mode = os.getenv("LLAMACU_DEBUG", "0").lower() in ("1", "true", "yes")
    perf_mode = os.getenv("LLAMACU_PERF", "0").lower() in ("1", "true", "yes")
    
    # 公共编译参数
    common_cxx_args = ["-std=c++17"]
    common_nvcc_args = [
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    
    if debug_mode:
        print("Debug mode enabled (LLAMACU_DEBUG=1)")
        cxx_args = common_cxx_args + [
            "-g3",           # 最详细的调试信息
            "-O0",           # 禁用优化
            "-DDISABLE_MEMPOOL",
            "-DDEBUG", 
            "-fno-inline",   # 禁用内联
            "-fno-omit-frame-pointer",  # 保留栈帧指针
        ]
        nvcc_base_args = common_nvcc_args + [
            "-O0", 
            "-g",            # 主机端调试信息
            "-lineinfo",     # 生成行号信息
            "-DDISABLE_MEMPOOL",
            "-DDEBUG", 
            "-DCUDA_DEBUG",
            "-Xcompiler", "-g3",              # 传递给主机编译器
            "-Xcompiler", "-fno-inline",      # 禁用内联
            "-Xcompiler", "-fno-omit-frame-pointer",  # 保留栈帧指针
        ]
        # Debug模式下的链接参数
        link_args = ["-g", "-rdynamic"]
    else:
        print("Release mode enabled")
        cxx_args = common_cxx_args + ["-O3"]
        nvcc_base_args = common_nvcc_args + [
            "-O3",
            "--use_fast_math",
        ]
        # Release模式下的链接参数
        link_args = []
    
    # 添加性能测试控制
    if perf_mode:
        print("Performance monitoring enabled (LLAMACU_PERF=1)")
        cxx_args.append("-DENABLE_PERF")
        nvcc_base_args.append("-DENABLE_PERF")
    else:
        print("Performance monitoring disabled (LLAMACU_PERF=0)")
    
    return cxx_args, nvcc_base_args, link_args

def get_all_headers():
    """获取所有头文件，用于依赖跟踪"""
    header_patterns = [
        "src/**/*.h",
        "src/**/*.hpp", 
        "src/**/*.cuh",
        "src/cutlass/include/**/*.h",
        "src/cutlass/include/**/*.hpp",
        "src/flash_attn/**/*.h",
        "src/flash_attn/**/*.hpp",
        "src/flash_attn/**/*.cuh",
    ]
    
    headers = []
    for pattern in header_patterns:
        abs_headers = glob.glob(os.path.join(this_dir, pattern), recursive=True)
        # 转换为相对路径
        rel_headers = [os.path.relpath(h, this_dir) for h in abs_headers]
        headers.extend(rel_headers)
    
    # 过滤掉不存在的文件（检查绝对路径但返回相对路径）
    headers = [h for h in headers if os.path.exists(os.path.join(this_dir, h))]
    
    return headers

# 尝试构建扩展模块
ext_modules = []
cmdclass = {}

try:
    # 尝试导入torch相关模块
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    # 获取CUDA架构
    arch_list = detect_cuda_arch()
    
    # 获取编译参数
    cxx_args, nvcc_base_args, link_args = get_compile_args()
    
    # 获取头文件
    all_headers = get_all_headers()
    
    # 为每个架构生成gencode参数
    gencode_args = []
    arch_defines = []
    for arch in arch_list:
        gencode_args.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        arch_defines.append(f"-D_ARCH{arch}")
    
    print(f"Using CUDA architecture compile flags: {arch_list}")
    
    # 创建Ninja构建扩展类
    class NinjaBuildExtension(BuildExtension):
        def __init__(self, *args, **kwargs) -> None:
            # do not override env MAX_JOBS if already exists
            if not os.environ.get("MAX_JOBS"):
                import psutil
                # calculate the maximum allowed NUM_JOBS based on cores
                max_num_jobs_cores = max(1, os.cpu_count() // 2)
                # calculate the maximum allowed NUM_JOBS based on free memory
                free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
                max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4
                # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
                max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                os.environ["MAX_JOBS"] = str(max_jobs)
            super().__init__(*args, **kwargs)
    
    # 配置扩展模块
    ext_modules = [
        CUDAExtension(
            name='llamacu.C',
            sources = [
                "src/entry.cu",
                "src/utils.cu",
                "src/signal_handler.cu",
                "src/perf.cu",
                *glob.glob("src/qgemm/gptq_marlin/*cu"),
                # *glob.glob("src/flash_attn/src/*.cu"),
                # *glob.glob("src/flash_attn/src/*hdim64_fp16*.cu"),
                *glob.glob("src/flash_attn/src/*hdim128_fp16*.cu"),
                # *glob.glob("src/flash_attn/src/*hdim64_bf16*.cu"),
                *glob.glob("src/flash_attn/src/*hdim128_bf16*.cu"),
            ],
            libraries=["cublas", "dl"],
            depends=all_headers,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": append_nvcc_threads(
                    nvcc_base_args + 
                    gencode_args +
                    arch_defines + [
                        # 添加依赖文件生成选项
                        "-MMD", "-MP",
                    ]
                ),
            },
            extra_link_args=link_args,
            include_dirs=[
                f"{this_dir}/src/flash_attn",
                f"{this_dir}/src/flash_attn/src",
                f"{this_dir}/src/cutlass/include",
                f"{this_dir}/src/",
            ],
        )
    ]
    
    cmdclass = {'build_ext': NinjaBuildExtension}
    
except Exception as e:
    print(f"警告: 无法配置CUDA扩展模块: {e}")
    print("跳过扩展模块构建，仅安装Python包...")

setup(
    name='llamacu',
    version='0.0.0',
    author_email="acha131441373@gmail.com",
    description="llama cuda implementation",
    packages=find_packages(),
    setup_requires=[
        "pybind11",
        "psutil",
        "ninja",
        "torch",
    ],
    install_requires=[
        "transformers==4.46.2",
        "accelerate==0.26.0",
        "datasets",
        "fschat",
        "openai",
        "anthropic",
        "human_eval",
        "zstandard",
        "tree_sitter",
        "tree-sitter-python"
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
) 