#ifndef RANS_LINEAR_H_
#define RANS_LINEAR_H_

#include "../cuda/rans_host.cuh"
#include "../cuda/rans_linear_kernel.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <torch/extension.h>

torch::Tensor fused_rans_linear_cuda(at::Tensor x, at::Tensor exp_stream,
                                     at::Tensor mantissa, at::Tensor bias,
                                     at::Tensor states, at::Tensor sizes,
                                     uint32_t num_streams,
                                     at::Tensor checkpoints, at::Tensor tables,
                                     at::Tensor slot_map, at::Tensor output) {

    // Instantiate the custom RansConfig and Loader
    using RansConfig = RansConfig8;

    // 1. Dimensions
    int M = x.size(0);
    int K = x.size(1);
    int N = output.size(1); // Use output size for N consistency

    // 1. Types & Config
    using RansConfig = RansConfig8;
    using ElementA = cutlass::bfloat16_t;
    using ElementB = uint8_t; // The compressed byte type
    using ElementC = cutlass::bfloat16_t;

    // We define our custom Loader type
    using RansLoader = RansWeightLoader<RansConfig, 128, 128>;

    // 2. Define the Internal Kernel using DefaultGemm factory
    // This ensures parameters like 'Alignment' and 'Operator' are valid
    using DefaultGemm = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA, 1, // A
        ElementB, LayoutB, 1, // B
        ElementC, LayoutC,
        float, // Accumulator
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementC, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3              // Stages
        >::GemmKernel; // Note the .GemmKernel at the end

    // 3. Wrap it in the Universal Adapter
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<DefaultGemm>;

    // 4. Arguments (Use explicit struct names)
    // We pass our RansWeightLoader::Params in the 'ptr_B' slot
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
        1, // Batch count
        {(ElementA *)x.data_ptr<at::BFloat16>(), K},
        {(ElementB *)exp_stream.data_ptr<uint8_t>(),
         (uint8_t *)mantissa.data_ptr<uint8_t>(),
         (ElementB *)exp_stream.data_ptr<uint8_t>(), (uint32_t)N,
         (uint32_t *)checkpoints.data_ptr<uint32_t>()},
        {(ElementC *)bias.data_ptr<at::BFloat16>(), 0},
        {(ElementC *)output.data_ptr<at::BFloat16>(), N}, {1.0f, 1.0f});

    // 5. Execution
    Gemm gemm_op;
    auto status = gemm_op(args, nullptr, at::cuda::getCurrentCUDAStream());

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS error: ", cutlassGetStatusString(status));

    return output;
}
#endif // RANS_LINEAR_H_
