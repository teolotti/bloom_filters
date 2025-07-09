cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(bloom_filters LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(bloom_filters main.cpp main.cu
        utils/BloomFilters.h
        utils/cuda_bloom.cuh
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(bloom_filters PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
