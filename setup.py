# from setuptools import setup, find_packages
# import os
# import sys
# import subprocess
# import platform
# from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
# import torch

# # Ensure pybind11 is available
# try:
#     from pybind11.setup_helpers import (
#         Pybind11Extension,
#         build_ext as pybind11_build_ext,
#     )
#     import pybind11
# except ImportError:
#     print("Installing pybind11...")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.0.0"])
#     from pybind11.setup_helpers import (
#         Pybind11Extension,
#         build_ext as pybind11_build_ext,
#     )
#     import pybind11


# def check_cuda_available() -> bool:
#     """Check if CUDA (nvcc) is available, unless disabled via env."""
#     if os.environ.get("DISABLE_CUDA", "0") == "1":
#         print("🔧 CUDA explicitly disabled by environment variable (DISABLE_CUDA=1).")
#         return False
#     try:
#         subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
#         return True
#     except (OSError, subprocess.CalledProcessError):
#         return False


# class CUDAExtension(Pybind11Extension):
#     """Extension that may include CUDA sources."""

#     def __init__(self, name, sources, *args, **kwargs):
#         self.cuda_sources = [s for s in sources if s.endswith(".cu")]
#         cpp_sources = [s for s in sources if not s.endswith(".cu")]
#         super().__init__(name, cpp_sources, *args, **kwargs)


# class BuildExtOptionalCUDA(pybind11_build_ext):
#     """Custom build_ext that optionally builds with CUDA."""

#     def run(self):
#         cuda_available = check_cuda_available()
#         if cuda_available:
#             print("✅ CUDA detected — building with GPU support.")
#         else:
#             print("⚠️ Building CPU-only version.")
#         for ext in self.extensions:
#             self._process_extension(ext, cuda_available)
#         super().run()

#     def _process_extension(self, ext, cuda_available: bool):
#         """Modify build process based on CUDA availability."""
#         if cuda_available:
#             ext.define_macros = ext.define_macros or []
#             ext.define_macros.append(("USE_CUDA", "1"))
#         else:
#             ext.define_macros = ext.define_macros or []
#             ext.define_macros.append(("USE_CUDA", "0"))

#         if not cuda_available or not getattr(ext, "cuda_sources", []):
#             return

#         build_dir = os.path.join("build", "cuda_objects")
#         os.makedirs(build_dir, exist_ok=True)
#         cuda_objects = []

#         import sysconfig
#         import torch
#         from torch.utils import cpp_extension

#         # Add PyTorch include paths
#         TORCH_INCLUDES = cpp_extension.include_paths()
#         PYTHON_INCLUDE = sysconfig.get_paths()["include"]

#         nvcc_flags = [
#             "-O3",
#             "-std=c++17",
#             "-gencode=arch=compute_61,code=sm_61",
#             "-gencode=arch=compute_75,code=sm_75",
#             "-gencode=arch=compute_80,code=sm_80",
#             "--compiler-options",
#             "-fPIC",
#         ] + [f"-I{inc}" for inc in TORCH_INCLUDES + [PYTHON_INCLUDE]]

#         for cu_src in ext.cuda_sources:
#             obj_path = os.path.join(build_dir, os.path.basename(cu_src) + ".o")
#             cmd = ["nvcc"] + nvcc_flags + ["-c", cu_src, "-o", obj_path]
#             print("Compiling CUDA source:", " ".join(cmd))
#             subprocess.check_call(cmd)
#             cuda_objects.append(obj_path)

#         ext.extra_objects = getattr(ext, "extra_objects", []) + cuda_objects
#         ext.libraries = getattr(ext, "libraries", []) + ["cudart"]

#         cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
#         if os.path.exists(os.path.join(cuda_home, "include")):
#             ext.include_dirs.append(os.path.join(cuda_home, "include"))
#         if platform.system() == "Windows":
#             cuda_lib_dir = os.path.join(cuda_home, "lib", "x64")
#         else:
#             cuda_lib_dir = os.path.join(cuda_home, "lib64")
#         if os.path.exists(cuda_lib_dir):
#             ext.library_dirs.append(cuda_lib_dir)


# # -------------------------------
# # Extension definition
# # -------------------------------
# sources = [
#     "src/python_bindings.cpp",
#     "src/cpp/wrapper.cpp",
#     "src/cuda/kernels.cu",  # Optional CUDA source
#     "src/cpp/compressed_linear_bindings.cpp",
#     "src/cuda/compressed_linear.cu",
# ]

# core_module = CUDAExtension(
#     name="comp_inference._core",
#     sources=sources,
#     include_dirs=[
#         "src",
#         pybind11.get_include(),
#         torch.utils.cpp_extension.include_paths(),
#     ],
#     extra_compile_args=["-std=c++17"],
#     language="c++",
# )


# # -------------------------------
# # Setup configuration
# # -------------------------------
# setup(
#     name="comp_inference",
#     version="0.1.0",
#     packages=find_packages(),
#     ext_modules=[core_module],
#     cmdclass={"build_ext": BuildExtOptionalCUDA},
#     package_data={"comp_inference": ["_core*.so"]},
#     install_requires=[
#         "torch",
#         "pybind11>=2.0.0",
#     ],
#     python_requires=">=3.8",
#     description="Compressed inference framework with optional CUDA acceleration.",
# )


from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import torch
import os

# Detect if CUDA is available
use_cuda = torch.cuda.is_available() and os.environ.get("DISABLE_CUDA", "0") != "1"
print(
    "✅ CUDA detected"
    if use_cuda
    else "⚠️ CUDA not detected, building CPU-only version."
)

# Source files
cpp_sources = [
    "src/python_bindings.cpp",
    "src/cpp/wrapper.cpp",
]

cuda_sources = [
    "src/cuda/kernels.cu",
    "src/cuda/compressed_linear.cu",
    "src/cuda/rans_ops.cu",
]

sources = cpp_sources + (cuda_sources if use_cuda else [])

# Select extension type
extension_class = CUDAExtension if use_cuda else CppExtension

ext_modules = [
    extension_class(
        name="comp_inference._core",
        sources=sources,
        include_dirs=["src"],  # Add your local includes
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17"] if use_cuda else [],
        },
    )
]

setup(
    name="comp_inference",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pybind11>=2.0.0",
    ],
    python_requires=">=3.8",
    description="Compressed inference framework with optional CUDA acceleration.",
)
