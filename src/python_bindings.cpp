#include "cpp/wrapper.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for CUDA accelerated operations";

    m.def(
        "vector_add",
        [](py::array_t<float> a, py::array_t<float> b) {
            // Get array info
            py::buffer_info a_info = a.request();
            py::buffer_info b_info = b.request();

            if (a_info.ndim != 1 || b_info.ndim != 1)
                throw std::runtime_error("Arrays must be 1D");

            if (a_info.size != b_info.size)
                throw std::runtime_error("Arrays must have same size");

            // Allocate output array
            size_t size = a_info.size;
            py::array_t<float> result = py::array_t<float>(size);
            py::buffer_info r_info = result.request();

            // Call the C++ wrapper which calls CUDA
            cuda_vector_add(static_cast<float *>(a_info.ptr),
                            static_cast<float *>(b_info.ptr),
                            static_cast<float *>(r_info.ptr), size);

            return result;
        },
        "Add two vectors using CUDA");
}
