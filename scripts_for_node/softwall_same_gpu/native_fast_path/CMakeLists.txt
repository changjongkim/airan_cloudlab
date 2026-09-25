cmake_minimum_required(VERSION 3.20)
project(softwall_native_fixture_check LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_executable(softwall_native_fixture_check native_fixture_check.cpp)
target_link_libraries(softwall_native_fixture_check PRIVATE OpenSSL::Crypto)
target_compile_options(softwall_native_fixture_check PRIVATE -Wall -Wextra -Werror)
