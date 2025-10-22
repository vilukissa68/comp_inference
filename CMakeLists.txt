cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

option(USE_CUDA10 "Build with CUDA 10.x toolkit instead of system default" OFF)

# CUDA Toolkit selection
if (USE_CUDA10)
    message(STATUS "Building with CUDA 10.x Toolkit (/usr/local/cuda-10.2)")
	set(CMAKE_CXX_COMPILER /opt/gcc-8/bin/g++)
    set(CMAKE_CUDA_HOST_COMPILER /opt/gcc-8/bin/gcc)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.2" CACHE PATH "Path to CUDA 10 toolkit")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
else()
    message(STATUS "Building with default CUDA Toolkit (/usr/local/cuda)")
	set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8" CACHE PATH "Path to CUDA 12 toolkit")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()

# Compiler and standard configuration
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDA REQUIRED)

# Build targets
add_library(comp_inference SHARED
    src/cuda/kernels.cu
    src/cpp/wrapper.cpp
)
target_include_directories(comp_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

add_executable(simulate_decompression_test
    tests/simulate_decompression.cpp
)
target_include_directories(simulate_decompression_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
)
target_link_libraries(simulate_decompression_test PRIVATE comp_inference)

# Optional: Python bindings
# find_package(pybind11 CONFIG REQUIRED)
# pybind11_add_module(_core src/python_bindings.cpp)
# target_link_libraries(_core PRIVATE comp_inference)
# install(TARGETS _core DESTINATION python/comp_inference)
