from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

sources = [
    'src/python_bindings.cpp',
    'src/cuda/rans_ops.cu', 
]

setup(
    name='comp_inference',
    version='0.1.0',
    packages=find_packages(),  # This finds the 'comp_inference' python folder
    ext_modules=[
        CUDAExtension(
            name='comp_inference.ccore',
            sources=sources,
            include_dirs=[
                os.path.abspath('src/cpp'),
                os.path.abspath('src/cuda'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', 
                    '--use_fast_math',
                    '-std=c++17'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch==2.9.1',
        'numpy<=2.2',
        'transformers' # Needed for your high-level tests
    ]
)
