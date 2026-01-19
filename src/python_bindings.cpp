#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 

// FIX: Include CUDA Runtime directly here
#include <cuda_runtime.h> 

#include "cpp/rans.hpp" 

namespace py = pybind11;
using namespace pybind11::literals;

struct TensorCompressResult {
    bool success;
    torch::Tensor stream;       // uint8
    torch::Tensor states;       // int32
    torch::Tensor output_sizes; // int32
    uint32_t num_streams;
    size_t stream_len;
};

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

	m.def("allocate_pinned_tensor", [](size_t size) {
        void* ptr = nullptr;
        
        // 1. Allocate raw pinned memory
        if (cudaMallocHost(&ptr, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned memory");
        }

        // 2. Define a deleter (lambda) that PyTorch calls when the Tensor dies
        auto deleter = [ptr](void*) {
            cudaFreeHost(ptr);
        };

        // 3. Define Tensor Options
        // CRITICAL: .pinned_memory(true) sets the flag so PyTorch knows 
        // it can use DMA for async transfers.
        auto options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .layout(torch::kStrided)
            .device(torch::kCPU) // Pinned memory is technically CPU memory
            .pinned_memory(true); 

        // 4. Create Tensor from raw pointer
        // {static_cast<long>(size)} is the shape (1D)
        return torch::from_blob(ptr, {static_cast<long>(size)}, deleter, options);
    }, "Allocates a pinned PyTorch Tensor");

    // Register the Tensor-based struct for Python
    py::class_<TensorCompressResult>(m, "TensorCompressResult")
        .def_readonly("success", &TensorCompressResult::success)
        .def_readonly("stream", &TensorCompressResult::stream)
        .def_readonly("states", &TensorCompressResult::states)
        .def_readonly("output_sizes", &TensorCompressResult::output_sizes)
        .def_readonly("num_streams", &TensorCompressResult::num_streams)
        .def_readonly("stream_len", &TensorCompressResult::stream_len);

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
                            py::array_t<uint16_t> cdf) -> RansManager::CompressResult {
		    std::cout << "Using numpy arrays for compress()" << std::endl;
            auto d = data.request();
            auto f = freqs.request();
            auto c = cdf.request();
            return self.compress((uint8_t*)d.ptr, d.size, (uint16_t*)f.ptr, (uint16_t*)c.ptr);
        })

        // 3. The Compress Binding
        .def("compress", [](RansManager& self, 
                            at::Tensor data, 
                            at::Tensor freqs, 
                            at::Tensor cdf) -> TensorCompressResult {
            
            auto res = self.compress(
                data.data_ptr<uint8_t>(), 
                data.numel(), 
                freqs.data_ptr<uint16_t>(), 
                cdf.data_ptr<uint16_t>()
            );

            auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8);
            auto stream_t = torch::from_blob(
                res.stream.data(), 
                {static_cast<long>(res.stream.size())}, 
                opts_u8
            ).clone();

            auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32);
            
            auto states_t = torch::from_blob(
                res.states.data(), 
                {static_cast<long>(res.states.size())}, 
                opts_i32
            ).clone();

            auto sizes_t = torch::from_blob(
                res.sizes.data(), 
                {static_cast<long>(res.sizes.size())}, 
                opts_i32
            ).clone();

            return TensorCompressResult{
                res.success,
                stream_t,
                states_t,
                sizes_t,
                res.num_streams,
                res.stream_len
            };
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
	})

	.def("decompress", [](RansManager& self,
								at::Tensor stream,
								at::Tensor states,
								at::Tensor sizes,
								uint32_t num_streams,
								size_t output_len,
								at::Tensor freqs,
								at::Tensor cdf) {

				// Allocate Torch Tensor on the same device as input stream
				auto options = torch::TensorOptions().dtype(torch::kUInt8).device(stream.device());
				at::Tensor output = torch::empty({(long)output_len}, options);

				float time_ms = self.decompress(
					stream.data_ptr<uint8_t>(), stream.numel(),
					states.data_ptr<uint32_t>(), sizes.data_ptr<uint32_t>(),
					num_streams, output_len,
					freqs.data_ptr<uint16_t>(), cdf.data_ptr<uint16_t>(),
					output.data_ptr<uint8_t>()
				);
				return py::make_tuple(output, time_ms);
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

	    std::cout << "Using numpy arrays decompress_into()" << std::endl;
		float time_ms = self.decompress(
			(uint8_t*)s_inf.ptr, s_inf.size,
			(uint32_t*)st_inf.ptr, (uint32_t*)sz_inf.ptr,
			num_streams, out_inf.size,
			(uint16_t*)f_inf.ptr, (uint16_t*)c_inf.ptr,
			(uint8_t*)out_inf.ptr 
		);
		return time_ms;
	})
	.def("decompress_into", [](RansManager& self,
								at::Tensor stream,
								at::Tensor states,
								at::Tensor sizes,
								uint32_t num_streams,
								at::Tensor freqs,
								at::Tensor cdf,
								at::Tensor output_buffer) {

	    std::cout << "Using Tensor decompress_into()" << std::endl;
		float time_ms = self.decompress(
			stream.data_ptr<uint8_t>(), stream.numel(),
            reinterpret_cast<uint32_t*>(states.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(sizes.data_ptr<int32_t>()),
			num_streams, output_buffer.numel(),
			freqs.data_ptr<uint16_t>(), cdf.data_ptr<uint16_t>(),
			output_buffer.data_ptr<uint8_t>()
		);
		return time_ms;
	});

}
