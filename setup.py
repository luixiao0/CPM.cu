from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='llamacu',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'llamacu.C',
            [
                'src/model.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 