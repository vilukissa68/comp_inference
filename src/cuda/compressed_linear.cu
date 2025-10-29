#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void linear_forward_kernel_fp32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_features, int out_features) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size * out_features) {
        int b = row / out_features;
        int o = row % out_features;
        float val = 0.0f;
        for (int i = 0; i < in_features; i++) {
            val += input[b * in_features + i] * weight[o * in_features + i];
        }
        if (bias != nullptr)
            val += bias[o];
        output[row] = val;
    }
}

// __global__ void linear_forward_kernel_bf16(
// 	const __nv_bfloat16* __restrict__ input,
// 	const __nv_bfloat16* __restrict__ weight,
// 	const __nv_bfloat16* __restrict__ bias,
// 	__nv_bfloat16* __restrict__ output,
// 	int batch_size, int in_features, int out_features) {
	
// 	int row = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < batch_size * out_features) {
// 		int b = row / out_features; // Output batch
// 		int o = row % out_features; // Output element
// 		__nv_bfloat16 val = __float2bfloat16(0.0f); // Accumulator
// 		for (int i = 0; i < in_features; i++) {
// 			val = __hadd(val, __hmul(input[b * in_features + i], weight[o * in_features + i]));
// 		}
// 		if (bias != nullptr)
// 			val = __hadd(val, bias[o]);
// 		output[row] = val;
// 	}
// }

// accumulate in float for accuracy, convert back to bfloat16 at the end
__global__ void linear_forward_kernel_bf16(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int in_features, int out_features) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (row < total) {
        int b = row / out_features; // batch index
        int o = row % out_features; // output feature index

        float sum = 0.0f; // accumulate in fp32
        int input_base = b * in_features;
        int weight_base = o * in_features;

		// dot product
        for (int i = 0; i < in_features; ++i) {
			// accumulate in fp32
            float a = __bfloat162float(input[input_base + i]); 
            float w = __bfloat162float(weight[weight_base + i]);
            sum += a * w; // MAC
        }

		// bias in fp32
        if (bias != nullptr) {
            sum += __bfloat162float(bias[o]);
        }

		// convert back to bfloat16
        output[row] = __float2bfloat16(sum);
    }
}

torch::Tensor linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());

    int threads = 256;
    int blocks = (batch_size * out_features + threads - 1) / threads;

	if (input.scalar_type() == torch::kBFloat16) {
		linear_forward_kernel_bf16<<<blocks, threads>>>(
			reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
			reinterpret_cast<__nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
			bias.defined() ? reinterpret_cast<__nv_bfloat16*>(bias.data_ptr<at::BFloat16>()) : nullptr,
			reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
			batch_size, in_features, out_features
		);
		return output;
	}
    linear_forward_kernel_fp32<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );

    return output;
}
