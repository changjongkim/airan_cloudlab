#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kSubcarriers = 3276;
constexpr std::size_t kSymbols = 14;
constexpr std::size_t kAntennas = 4;
constexpr std::size_t kElements = kSubcarriers * kSymbols * kAntennas;

struct CudaArray {
    void* pointer{};
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    std::string typestr;
};

CudaArray inspect(const py::object& value, const char* label) {
    if (!py::hasattr(value, "__cuda_array_interface__")) {
        throw std::invalid_argument(std::string(label) + " has no CUDA array interface");
    }
    const py::dict interface = value.attr("__cuda_array_interface__").cast<py::dict>();
    const int version = interface["version"].cast<int>();
    if (version < 2 || version > 3) {
        throw std::invalid_argument(std::string(label) + " has unsupported CUDA array version");
    }
    const py::tuple data = interface["data"].cast<py::tuple>();
    if (data.size() != 2 || data[1].cast<bool>()) {
        throw std::invalid_argument(std::string(label) + " must be writable device memory");
    }
    CudaArray result;
    result.pointer = reinterpret_cast<void*>(data[0].cast<std::uintptr_t>());
    result.shape = interface["shape"].cast<std::vector<std::size_t>>();
    result.typestr = interface["typestr"].cast<std::string>();
    if (interface.contains("strides") && !interface["strides"].is_none()) {
        result.strides = interface["strides"].cast<std::vector<std::size_t>>();
    }
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

bool is_float32(const std::string& typestr) {
    return typestr == "<f4" || typestr == "=f4" || typestr == "|f4";
}

bool is_complex64(const std::string& typestr) {
    return typestr == "<c8" || typestr == "=c8" || typestr == "|c8";
}

__global__ void assemble_iq_kernel(
    const float* __restrict__ planes,
    float2* __restrict__ slot) {
    const std::size_t c_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_index >= kElements) {
        return;
    }
    const std::size_t antenna = c_index % kAntennas;
    const std::size_t quotient = c_index / kAntennas;
    const std::size_t symbol = quotient % kSymbols;
    const std::size_t subcarrier = quotient / kSymbols;
    const std::size_t fortran_index =
        subcarrier + kSubcarriers * (symbol + kSymbols * antenna);
    slot[fortran_index] = make_float2(planes[c_index], planes[kElements + c_index]);
}

void assemble_iq(const py::object& planes_object,
                 const py::object& slot_object,
                 std::uintptr_t stream_value) {
    const CudaArray planes = inspect(planes_object, "planes");
    const CudaArray slot = inspect(slot_object, "slot");
    require(is_float32(planes.typestr), "planes must have float32 dtype");
    require(planes.shape == std::vector<std::size_t>{2 * kElements},
            "planes must be a flat two-plane array with 366912 entries");
    require(planes.strides.empty() || planes.strides == std::vector<std::size_t>{4},
            "planes must be C-contiguous");
    require(is_complex64(slot.typestr), "slot must have complex64 dtype");
    require(slot.shape == std::vector<std::size_t>{kSubcarriers, kSymbols, kAntennas},
            "slot shape must be (3276, 14, 4)");
    const std::vector<std::size_t> expected_strides{
        sizeof(float2),
        sizeof(float2) * kSubcarriers,
        sizeof(float2) * kSubcarriers * kSymbols,
    };
    require(slot.strides == expected_strides, "slot must be Fortran-contiguous complex64");
    require(planes.pointer != nullptr && slot.pointer != nullptr, "null CUDA pointer");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((kElements + threads - 1) / threads);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    assemble_iq_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(planes.pointer),
        static_cast<float2*>(slot.pointer));
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("assemble_iq_kernel launch failed: ") + cudaGetErrorString(status));
    }
}

}  // namespace

PYBIND11_MODULE(_softwall_native_iq, module) {
    module.doc() = "SoftWall exact raw-IQ plane to cuPHY slot bridge";
    module.def(
        "assemble_iq",
        &assemble_iq,
        py::arg("planes"),
        py::arg("slot"),
        py::arg("stream"),
        "Queue exact two-plane C-order to complex64 Fortran-order assembly.");
}
