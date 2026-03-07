#!/usr/bin/env python3

import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#     ],
#     key=["M", "N", "K"],
# )
# @triton.jit
# def matmul_kernel(
#     a_ptr,
#     b_ptr,
#     c_ptr,
#     M,
#     N,
#     K,
#     stride_am,
#     stride_ak,
#     stride_bk,
#     stride_bn,
#     stride_cm,
#     stride_cn,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
#         accumulator = tl.dot(a, b, accumulator)
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     c = accumulator.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# def triton_matmul(a, b):
#     a_shape = a.shape
#     a_flat = a.view(-1, a_shape[-1])
#     M, K = a_flat.shape
#     K, N = b.shape
#     c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

#     grid = lambda META: (
#         triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
#     )
#     matmul_kernel[grid](
#         a_flat,
#         b,
#         c,
#         M,
#         N,
#         K,
#         a_flat.stride(0),
#         a_flat.stride(1),
#         b.stride(0),
#         b.stride(1),
#         c.stride(0),
#         c.stride(1),
#     )
#     return c.view(*a_shape[:-1], N)


import torch
import triton
import triton.language as tl


# ==============================================================================
# 1. SPLIT-K GEMV KERNEL (For M=1 / Decode)
# ==============================================================================
@triton.jit
def gemv_split_k_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    K,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_k
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_idx = offs_k + k * BLOCK_SIZE_K * SPLIT_K

        a = tl.load(a_ptrs + k * BLOCK_SIZE_K * SPLIT_K, mask=k_idx < K, other=0.0)
        b = tl.load(
            b_ptrs + k * BLOCK_SIZE_K * SPLIT_K * stride_bk,
            mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.sum(a[:, None] * b, axis=0)

    c_ptrs = c_ptr + pid_k * N + offs_n
    tl.store(c_ptrs, accumulator, mask=offs_n < N)


def triton_gemv(a_flat, b):
    # a_flat: [K], b: [K, N]
    K, N = b.shape
    SPLIT_K = 8

    workspace = torch.empty((SPLIT_K, N), device=a_flat.device, dtype=torch.float32)

    # Hardcoded optimal GEMV config
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    # Grid explicitly bypasses META dependency
    grid = (triton.cdiv(N, BLOCK_SIZE_N), SPLIT_K)

    gemv_split_k_kernel[grid](
        a_flat,
        b,
        workspace,
        N,
        K,
        b.stride(0),
        b.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=SPLIT_K,
        num_warps=4,
        num_stages=3,
    )

    c = torch.sum(workspace, dim=0).to(torch.bfloat16)
    return c


# ==============================================================================
# 2. STANDARD BLOCK GEMM KERNEL (For M>1 / Prefill)
# ==============================================================================
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm(a_flat, b):
    M, K = a_flat.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a_flat.device, dtype=torch.bfloat16)

    # Hardcoded robust GEMM config
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    gemm_kernel[grid](
        a_flat,
        b,
        c,
        M,
        N,
        K,
        a_flat.stride(0),
        a_flat.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=3,
    )
    return c


# ==============================================================================
# 3. THE SMART ROUTER
# ==============================================================================
def triton_matmul(a, b):
    """Routes to GEMV for decode (M=1) and GEMM for prefill (M>1)"""
    a_shape = a.shape
    a_flat = a.view(-1, a_shape[-1])
    M, K = a_flat.shape

    if M == 1:
        # Use our fast Split-K Vector kernel (no recursion!)
        out = triton_gemv(a_flat, b)
    else:
        # Use the standard Block GEMM kernel (no recursion!)
        out = triton_gemm(a_flat, b)

    return out.view(*a_shape[:-1], b.shape[1])
