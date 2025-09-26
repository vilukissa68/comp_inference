#!/usr/bin/env python3

import ctypes
import sys

try:
    # Try to load CUDA libraries
    cuda = ctypes.CDLL("libcuda.so.1")
    cudart = ctypes.CDLL("libcudart.so")
    print("CUDA libraries loaded successfully")

    # Check CUDA device
    device_count = ctypes.c_int()
    cuda.cuInit(0)
    result = cudart.cudaGetDeviceCount(ctypes.byref(device_count))

    print(f"CUDA initialization result: {result}")
    print(f"CUDA device count: {device_count.value}")

    # Get driver version
    version = ctypes.c_int()
    result = cuda.cuDriverGetVersion(ctypes.byref(version))
    print(f"CUDA driver version: {version.value//1000}.{(version.value%100)//10}")
except Exception as e:
    print(f"Error: {e}")
