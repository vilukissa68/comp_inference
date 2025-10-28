
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cpp/wrapper.hpp"  // Your CUDA helper functions

namespace py = pybind11;

// -----------------------------
// linear_forward
// -----------------------------
torch::Tensor linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);  // CUDA implementation declared elsewhere

torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined())
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    return linear_forward_cuda(input, weight, bias);
}

// -----------------------------
// _core module
// -----------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for CUDA accelerated operations";

    // Linear forward
    m.def("linear_forward", &linear_forward, "Custom Linear forward (CUDA)");

    // Vector add
    m.def(
        "vector_add",
        [](py::array_t<float> a, py::array_t<float> b) {
            py::buffer_info a_info = a.request();
            py::buffer_info b_info = b.request();

            if (a_info.ndim != 1 || b_info.ndim != 1)
                throw std::runtime_error("Arrays must be 1D");
            if (a_info.size != b_info.size)
                throw std::runtime_error("Arrays must have same size");

            size_t size = a_info.size;
            py::array_t<float> result(size);
            py::buffer_info r_info = result.request();

            cuda_vector_add(
                static_cast<float *>(a_info.ptr),
                static_cast<float *>(b_info.ptr),
                static_cast<float *>(r_info.ptr),
                size
            );

            return result;
        },
        "Add two vectors using CUDA"
    );

    // Simulate decompression
    m.def(
        "simulate_decompression",
        [](py::array_t<uint8_t> input, uint64_t output_size) {
            py::buffer_info in_info = input.request();
            if (in_info.ndim != 1)
                throw std::runtime_error("Arrays must be 1D");

            py::array_t<uint8_t> output(output_size);
            py::buffer_info out_info = output.request();

            float decomp_time_ms = simulate_decompression(
                static_cast<uint8_t *>(in_info.ptr),
                static_cast<uint8_t *>(out_info.ptr),
                in_info.size,
                output_size
            );

            return py::make_tuple(output, decomp_time_ms);
        },
        "Simulate decompression using CUDA"
    );
}
