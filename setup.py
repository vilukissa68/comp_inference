from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import platform
import pkg_resources

# Ensure pybind11 3.0 is installed
try:
    pkg_resources.require("pybind11>=3.0.0")
except pkg_resources.DistributionNotFound:
    print("pybind11 3.0 not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=3.0.0"])

# Import get_pybind_include after ensuring correct version is installed
try:
    # Get pybind11 include path programmatically from the installed version
    from pybind11.setup_helpers import (
        Pybind11Extension,
        build_ext as pybind11_build_ext,
    )
    import pybind11
except ImportError:
    raise ImportError("Failed to import pybind11. Make sure it's installed correctly.")


class CUDAExtension(Pybind11Extension):
    def __init__(self, name, sources, *args, **kwargs):
        Pybind11Extension.__init__(self, name, sources, *args, **kwargs)
        # Save CUDA sources for special handling
        self.cuda_sources = [src for src in sources if src.endswith(".cu")]
        # Remove CUDA sources from regular sources
        self.sources = [src for src in sources if not src.endswith(".cu")]


class BuildExtWithCUDA(pybind11_build_ext):
    def run(self):
        cuda_available = self._check_cuda_available()
        if not cuda_available:
            raise RuntimeError("CUDA is required to build this extension")

        # Process the extensions
        for ext in self.extensions:
            self._add_cuda_flags(ext)
            self._compile_cuda_files(ext)

        # Continue with regular build
        pybind11_build_ext.run(self)

    def _check_cuda_available(self):
        """Check if CUDA is available"""
        try:
            output = subprocess.check_output(["nvcc", "--version"])
            return True
        except:
            return False

    def _add_cuda_flags(self, ext):
        """Add CUDA compilation flags"""
        # Find CUDA include directory
        cuda_include = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        if not os.path.exists(os.path.join(cuda_include, "include")):
            cuda_include = "/usr/local/cuda"

        # Add CUDA include directory
        if os.path.exists(os.path.join(cuda_include, "include")):
            ext.include_dirs.append(os.path.join(cuda_include, "include"))

        # Add CUDA library directory
        if platform.system() == "Windows":
            cuda_lib_dir = os.path.join(cuda_include, "lib", "x64")
        else:
            cuda_lib_dir = os.path.join(cuda_include, "lib64")

        if os.path.exists(cuda_lib_dir):
            ext.library_dirs.append(cuda_lib_dir)

        # Add CUDA runtime library
        ext.libraries.append("cudart")

    def _compile_cuda_files(self, ext):
        """Compile CUDA files and add output objects to sources"""
        if not ext.cuda_sources:
            return

        import time

        temp_dir = "temp_" + str(int(time.time()))

        # Create a build directory for CUDA objects
        build_dir = os.path.join("build", temp_dir)
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # Set NVCC flags
        nvcc_flags = [
            "-O3",
            # Compile for multiple architectures
            "-gencode=arch=compute_60,code=sm_60",  # Pascal (GTX 1080)
            "-gencode=arch=compute_61,code=sm_61",  # Pascal (GTX 1080 Ti)
            "-gencode=arch=compute_70,code=sm_70",  # Volta (V100)
            "-gencode=arch=compute_75,code=sm_75",  # Turing (RTX 2080)
            # Add PTX code for future compatibility
            "-gencode=arch=compute_61,code=compute_61",
            "--compiler-options",
            "'-fPIC'",
        ]
        # Compile each CUDA source file
        cuda_objects = []
        for cuda_source in ext.cuda_sources:
            # Get the filename without extension
            file_basename = os.path.splitext(os.path.basename(cuda_source))[0]
            # Define the output object file
            output_file = os.path.join(build_dir, file_basename + ".o")
            cuda_objects.append(output_file)

            # Compile the CUDA file
            command = ["nvcc"] + nvcc_flags + ["-c", cuda_source, "-o", output_file]
            print("Compiling CUDA:", " ".join(command))
            subprocess.check_call(command)

        # Add the compiled objects to the sources
        ext.extra_objects = cuda_objects


# Define the extension
core_module = CUDAExtension(
    "comp_inference._core",
    sources=[
        "src/python_bindings.cpp",
        "src/cpp/wrapper.cpp",
        "src/cuda/kernels.cu",  # This will be handled specially
    ],
    include_dirs=[
        "src",
        pybind11.get_include(),  # Add pybind11 include path explicitly
    ],
    extra_compile_args=["-std=c++14"],
)

setup(
    name="comp_inference",
    version="0.1",
    packages=["comp_inference"],
    ext_modules=[core_module],
    cmdclass={"build_ext": BuildExtWithCUDA},
    package_data={"comp_inference": ["_core*.so"]},
    install_requires=[
        "pybind11>=3.0.0",  # Explicitly require pybind11 version 3.0 or higher
    ],
)
