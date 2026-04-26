cmake_minimum_required(VERSION 3.16)
project(bench_cpp CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(bench bench.cpp)

target_compile_options(bench PRIVATE
    -O2
    -march=native
)