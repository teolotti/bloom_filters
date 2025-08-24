#include <cuda_runtime.h>
#include <iostream>
#include "utils/cuda_bloom.cuh"


__global__ void bloom_insert_kernel(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    for (int i = 0; i < k; ++i) {
        size_t h = 0; 
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; 
        h = (h + seeds[i]) % m;
        atomicOr(&bit_array[h / 32], 1u << (h % 32));
    }
}

__global__ void bloom_query_kernel(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    bool possibly_present = true;
    for (int i = 0; i < k && possibly_present; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; 
        h = (h + seeds[i]) % m;
        if (!(bit_array[h / 32] & (1u << (h % 32)))) possibly_present = false;
    }
    results[idx] = possibly_present;
}

__global__ void bloom_insert_shared(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        shared_bits[i] = 0; 
    }

    if (idx < num_elems) {
        for (int i = 0; i < k; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j) 
                h = (h * 31 + elements[idx * elem_size + j]) % m; 
            h = (h + seeds[i]) % m; 
            atomicOr(&shared_bits[h / 32], 1u << (h % 32)); 
        }
    }
    __syncthreads();

    // Merge in globale
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        atomicOr(&bit_array[i], shared_bits[i]); 
    }
}

__global__ void bloom_query_shared(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;


    
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        shared_bits[i] = bit_array[i]; 
    }

    if (idx < num_elems) {
        bool possibly_present = true;
        for (int i = 0; i < k && possibly_present; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
                h = (h * 31 + elements[idx * elem_size + j]) % m; 
            h = (h + seeds[i]) % m;
            if (!(shared_bits[h / 32] & (1u << (h % 32)))) {
                possibly_present = false; 
            }
        }
        results[idx] = possibly_present;
    }
} 


__constant__ int const_seeds[NUM_SEEDS];

__global__ void bloom_insert_constant(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    for (int i = 0; i < k; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m;
        h = (h + const_seeds[i]) % m; // Use a constant seed for each hash function
        atomicOr(&bit_array[h / 32], 1u << (h % 32));
    }
}

__global__ void bloom_query_constant(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    bool possibly_present = true;
    for (int i = 0; i < k && possibly_present; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; 
        h = (h + const_seeds[i]) % m; // Use a constant seed for each hash function
        if (!(bit_array[h / 32] & (1u << (h % 32)))) possibly_present = false;
    }
    results[idx] = possibly_present;
}

__global__ void bloom_insert_hybrid(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        shared_bits[i] = 0; // 
    }

    if (idx < num_elems) {
        for (int i = 0; i < k; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
                h = (h * 31 + elements[idx * elem_size + j]) % m; 
            h = (h + const_seeds[i]) % m;
            atomicOr(&shared_bits[h / 32], 1u << (h % 32));
        }
    }
    __syncthreads();

    
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        atomicOr(&bit_array[i], shared_bits[i]); 
    }
}

__global__ void bloom_query_hybrid(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k, bool* results) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        shared_bits[i] = bit_array[i]; 
    }

    if (idx < num_elems) {
        bool possibly_present = true;
        for (int i = 0; i < k && possibly_present; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
                h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
            h = (h + const_seeds[i]) % m;
            if (!(shared_bits[h / 32] & (1u << (h % 32)))) possibly_present = false;
        }
        results[idx] = possibly_present;
    }
} 

void set_seeds_const(const int* seeds, int k) {
    CUDA_CHECK(cudaMemcpyToSymbol(const_seeds, seeds, k * sizeof(int)));
}

__global__ void bloom_insert_with_shared_elements_buffer(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k) {
    extern __shared__ char shared_elements[];  

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= num_elems) return;

    // Copia la stringa in shared memory
    for (int i = 0; i < elem_size; ++i) {
        shared_elements[tid * elem_size + i] = elements[idx * elem_size + i];
    }
    __syncthreads();  

    
    for (int i = 0; i < k; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && shared_elements[tid * elem_size + j] != '\0'; ++j) {
            h = (h * 31 + shared_elements[tid * elem_size + j]) % m;
        }
        h = (h + seeds[i]) % m;
        atomicOr(&bit_array[h / 32], 1u << (h % 32));
    }
}


__global__ void bloom_query_with_shared_elements_buffer(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results) {
    extern __shared__ char shared_elements[];  

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= num_elems) return;

    
    for (int i = 0; i < elem_size; ++i) {
        shared_elements[tid * elem_size + i] = elements[idx * elem_size + i];
    }
    __syncthreads();

    bool possibly_present = true;
    for (int i = 0; i < k && possibly_present; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && shared_elements[tid * elem_size + j] != '\0'; ++j) {
            h = (h * 31 + shared_elements[tid * elem_size + j]) % m;
        }
        h = (h + seeds[i]) % m;
        if (!(bit_array[h / 32] & (1u << (h % 32)))) {
            possibly_present = false;
        }
    }

    results[idx] = possibly_present;
}
