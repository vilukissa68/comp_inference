#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 

// FIX: Include CUDA Runtime directly here
#include <cuda_runtime.h> 

#include "cpp/rans.hpp" 

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(ccore, m) {
    m.doc() = "Python bindings for CUDA accelerated operations";

    m.def("allocate_pinned_memory", [](size_t size) {
        void* ptr = nullptr;
        // Direct call to CUDA Runtime API
        if (cudaMallocHost(&ptr, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned memory");
        }
        
        py::capsule free_when_done(ptr, [](void* p) {
            cudaFreeHost(p);
        });

        return py::array_t<uint8_t>(
            {size}, {sizeof(uint8_t)}, (uint8_t*)ptr, free_when_done
        );
    }, "Allocates a pinned numpy array");

    py::class_<RansManager::CompressResult>(m, "CompressResult")
	  	.def_readonly("success", &RansManager::CompressResult::success)
        .def_property_readonly("stream", [](const RansManager::CompressResult& r) {
            return py::array_t<uint8_t>(r.stream.size(), r.stream.data());
        })
        .def_property_readonly("states", [](const RansManager::CompressResult& r) {
            return py::array_t<uint32_t>(r.states.size(), r.states.data());
        })
        .def_property_readonly("output_sizes", [](const RansManager::CompressResult& r) {
            return py::array_t<uint32_t>(r.sizes.size(), r.sizes.data());
        })
        .def_readonly("num_streams", &RansManager::CompressResult::num_streams)
        .def_readonly("stream_len", &RansManager::CompressResult::stream_len);

    py::class_<RansManager>(m, "RansManager")
        .def(py::init<size_t>(), "max_data_hint"_a = 0)
        
        .def("compress", [](RansManager& self, 
                            py::array_t<uint8_t> data, 
                            py::array_t<uint16_t> freqs, 
                            py::array_t<uint16_t> cdf) {
            auto d = data.request();
            auto f = freqs.request();
            auto c = cdf.request();
            return self.compress((uint8_t*)d.ptr, d.size, (uint16_t*)f.ptr, (uint16_t*)c.ptr);
        })

        .def("decompress_into", [](RansManager& self,
                                   py::array_t<uint8_t> stream,
                                   py::array_t<uint32_t> states,
                                   py::array_t<uint32_t> sizes,
                                   uint32_t num_streams,
                                   py::array_t<uint16_t> freqs,
                                   py::array_t<uint16_t> cdf,
                                   py::array_t<uint8_t> output_buffer) {

            auto s_inf = stream.request();
            auto st_inf = states.request();
            auto sz_inf = sizes.request();
            auto f_inf = freqs.request();
            auto c_inf = cdf.request();
            auto out_inf = output_buffer.request();

            float time_ms = self.decompress(
                (uint8_t*)s_inf.ptr, s_inf.size,
                (uint32_t*)st_inf.ptr, (uint32_t*)sz_inf.ptr,
                num_streams, out_inf.size,
                (uint16_t*)f_inf.ptr, (uint16_t*)c_inf.ptr,
                (uint8_t*)out_inf.ptr 
            );
            return time_ms;
        })

        .def("decompress", [](RansManager& self,
                              py::array_t<uint8_t> stream,
                              py::array_t<uint32_t> states,
                              py::array_t<uint32_t> sizes,
                              uint32_t num_streams,
                              size_t output_len,
                              py::array_t<uint16_t> freqs,
                              py::array_t<uint16_t> cdf) {
            
            auto s_inf = stream.request();
            auto st_inf = states.request();
            auto sz_inf = sizes.request();
            auto f_inf = freqs.request();
            auto c_inf = cdf.request();

            auto output = py::array_t<uint8_t>(output_len);
            auto out_inf = output.request();

            float time_ms = self.decompress(
                (uint8_t*)s_inf.ptr, s_inf.size,
                (uint32_t*)st_inf.ptr, (uint32_t*)sz_inf.ptr,
                num_streams, output_len,
                (uint16_t*)f_inf.ptr, (uint16_t*)c_inf.ptr,
                (uint8_t*)out_inf.ptr 
            );
            return py::make_tuple(output, time_ms);
        });
}
