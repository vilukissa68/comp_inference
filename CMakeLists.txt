# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

# Get pybind11 from the Python installation
set(Python_EXECUTABLE "python")  # Adjust if necessary
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)


# Find packages
find_package(CUDA REQUIRED)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 60)  # For RTX 20xx series

# Create CUDA library
add_library(cuda_backend SHARED
    src/cuda/kernels.cu
    src/cpp/wrapper.cpp
)
target_include_directories(cuda_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Python bindings
pybind11_add_module(_core
    src/python_bindings.cpp
)
target_link_libraries(_core PRIVATE cuda_backend)

# Installation
install(TARGETS _core DESTINATION python/comp_inference)
