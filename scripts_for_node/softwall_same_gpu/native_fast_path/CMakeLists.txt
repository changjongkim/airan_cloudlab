cmake_minimum_required(VERSION 3.20)
project(softwall_native_fast_path LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_executable(softwall_native_fixture_check native_fixture_check.cpp)
target_link_libraries(softwall_native_fixture_check PRIVATE OpenSSL::Crypto)
target_compile_options(softwall_native_fixture_check PRIVATE -Wall -Wextra -Werror)

option(SOFTWALL_BUILD_CUDA_BRIDGE "Build the CUDA/pybind native IQ bridge" OFF)
if(SOFTWALL_BUILD_CUDA_BRIDGE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_softwall_native_iq native_iq_bridge.cu)
    set_target_properties(
        _softwall_native_iq
        PROPERTIES
            CUDA_ARCHITECTURES 80
            CUDA_STANDARD 17
            CXX_STANDARD 17)
    target_link_libraries(_softwall_native_iq PRIVATE CUDA::cudart)
    target_compile_options(
        _softwall_native_iq
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
endif()
