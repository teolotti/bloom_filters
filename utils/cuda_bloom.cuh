#include <driver_types.h>

const int NUM_SEEDS = 5; // Numero massimo di semi per le funzioni hash


// Macro per gestione errori CUDA
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void bloom_insert_kernel(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k);
__global__ void bloom_query_kernel(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results);
__global__ void bloom_insert_shared(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, int segment_bits);
__global__ void bloom_query_shared(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results, int segment_bits);
__global__ void bloom_insert_constant(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k);
__global__ void bloom_query_constant(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k, bool* results);
__global__ void bloom_insert_hybrid(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k);
__global__ void bloom_query_hybrid(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k, bool* results);
void set_seeds_const(const int* seeds, int k);
__global__ void bloom_insert_with_shared_elements_buffer(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k);
__global__ void bloom_query_with_shared_elements_buffer(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results);