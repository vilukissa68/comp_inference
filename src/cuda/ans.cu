#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void ans_compress_bf16(
	const __nv_bfloat16* __restrict__ input,
	const uint16_t* __restrict__ cdf,
	const uint16_t* __restrict__ freq,
	const uint32_t N,
	uint8_t* __restrict__ output,
) {

	uint32_t x = 0;
	for (uint32_t i = 0; i < N; i++) {
		// load bfloat16 value
		__nv_bfloat16 val = input[i];
		// convert to uint16_t for indexing
		uint16_t symbol = __bfloat162uint16(val);

		// get frequency and cumulative frequency
		uint16_t f = freq[symbol];
		uint16_t cf = cdf[symbol];

		// ANS encoding step
		while (x >= (1 << 16) * f) {
			// output least significant byte
			output[i] = x & 0xFF;
			x >>= 8;
		}
		x = ((x / f) << 16) + (x % f) + cf;
	}

}


uint8_t* ans_compress_bf16_cuda(
	const at::Tensor& input,
	const at::Tensor& cdf,
	const at::Tensor& freq,
	const uint32_t N,
	at::Tensor& output
) {
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	ans_compress_bf16<<<blocks, threads>>>(
		reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
		reinterpret_cast<uint16_t*>(cdf.data_ptr<uint16_t>()),
		reinterpret_cast<uint16_t*>(freq.data_ptr<uint16_t>()),
		N,
		reinterpret_cast<uint8_t*>(output.data_ptr<uint8_t>())
	);

	return output.data_ptr<uint8_t>();
}
