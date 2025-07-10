#include <cuda_runtime.h>
#include <iostream>
#include "utils/cuda_bloom.cuh"


__global__ void bloom_insert_kernel(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    for (int i = 0; i < k; ++i) {
        size_t h = 0; // Use a different seed for each hash function
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function, elem_size is the size of each element
        h = (h + seeds[i]) % m;
        atomicOr(&bit_array[h / 32], 1 << (h % 32));
    }
}

__global__ void bloom_query_kernel(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    bool possibly_present = true;
    for (int i = 0; i < k && possibly_present; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
        h = (h + seeds[i]) % m;
        if (!(bit_array[h / 32] & (1 << (h % 32)))) possibly_present = false;
    }
    results[idx] = possibly_present;
}

__global__ void bloom_insert_shared(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, int segment_bits) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int segment_words = (segment_bits + 31) / 32; // Number of 32-bit integers needed to represent segment_bits
    int segment_offset = blockIdx.x * segment_bits;

    // Inizializza shared memory
    // Each thread block will use shared memory to accumulate bits
    // Only allocate enough space for the number of 32-bit integers needed ((m + 31) / 32)
    for (int i = tid; i < segment_words; i += blockDim.x) {
        shared_bits[i] = 0; // Initialize shared memory bits to 0
    }
    __syncthreads(); // Ensure all threads have initialized shared memory

    if (idx < num_elems) {
        for (int i = 0; i < k; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j) // Ensure we don't read beyond the element size
                h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
            h = (h + seeds[i]) % m; // Use a different seed for each hash function
            // Set the bit in shared memory
            int segment_id = h / segment_bits; // Determine which segment this hash falls into
            if (segment_id!= blockIdx.x) continue; // Only process hashes that belong to this segment
            int local_h = h % segment_bits; // Local hash within the segment
            atomicOr(&shared_bits[local_h / 32], 1u << (local_h % 32));
        }
    }
    __syncthreads();

    // Merge in globale
    for (int i = tid; i < segment_words; i += blockDim.x) {
        atomicOr(&bit_array[(segment_offset / 32) + i], shared_bits[i]); // Merge shared bits into global memory
    }
}

__global__ void bloom_query_shared(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k, bool* results, int segment_bits) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int segment_words = (segment_bits + 31) / 32; // Number of 32-bit integers needed to represent segment_bits
    int segment_offset = blockIdx.x * segment_bits;

    // Inizializza shared memory
    for (int i = tid; i < segment_words; i += blockDim.x) {
        shared_bits[i] = bit_array[(segment_offset / 32) + i]; // Copy global bits to shared memory
    }

    if (idx < num_elems) {
        bool possibly_present = true;
        for (int i = 0; i < k && possibly_present; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
                h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
            h = (h + seeds[i]) % m;
            int segment_id = h / segment_bits; // Determine which segment this hash falls into
            if (segment_id != blockIdx.x) {
                possibly_present = false; // If the hash does not belong to this segment, it cannot be present
                break; // No need to check further if we already know it's not present
            }
            int local_h = h % segment_bits; // Local hash within the segment

            if (!(shared_bits[local_h / 32] & (1 << (local_h % 32)))) possibly_present = false;
        }
        results[idx] = possibly_present;
    }
} // probably not useful and not necessary, but kept for experimentation

// seeds for hash functions can be defined as constants (prime numbers or random values)
__constant__ int const_seeds[NUM_SEEDS];

__global__ void bloom_insert_constant(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    for (int i = 0; i < k; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
        h = (h + const_seeds[i]) % m; // Use a constant seed for each hash function
        atomicOr(&bit_array[h / 32], 1 << (h % 32));
    }
}

__global__ void bloom_query_constant(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    bool possibly_present = true;
    for (int i = 0; i < k && possibly_present; ++i) {
        size_t h = 0;
        for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
            h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
        h = (h + const_seeds[i]) % m; // Use a constant seed for each hash function
        if (!(bit_array[h / 32] & (1 << (h % 32)))) possibly_present = false;
    }
    results[idx] = possibly_present;
}

__global__ void bloom_insert_hybrid(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        shared_bits[i] = 0; // Initialize shared memory bits to 0
    }

    if (idx < num_elems) {
        for (int i = 0; i < k; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
                h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
            h = (h + const_seeds[i]) % m;
            atomicOr(&shared_bits[h / 32], 1 << (h % 32));
        }
    }
    __syncthreads();

    // Merge in global memory
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        atomicOr(&bit_array[i], shared_bits[i]); // Merge shared bits into global memory
    }
}

__global__ void bloom_query_hybrid(uint32_t* bit_array, const char* elements, int elem_size, int num_elems, int m, int k, bool* results) {
    extern __shared__ uint32_t shared_bits[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    for (int i = tid; i < (m + 31) / 32; i += blockDim.x) {
        shared_bits[i] = bit_array[i]; // Copy global bits to shared memory
    }

    if (idx < num_elems) {
        bool possibly_present = true;
        for (int i = 0; i < k && possibly_present; ++i) {
            size_t h = 0;
            for (int j = 0; j < elem_size && elements[idx * elem_size + j] != '\0'; ++j)
                h = (h * 31 + elements[idx * elem_size + j]) % m; // Simple hash function
            h = (h + const_seeds[i]) % m;
            if (!(shared_bits[h / 32] & (1 << (h % 32)))) possibly_present = false;
        }
        results[idx] = possibly_present;
    }
} // probably not useful and not necessary, but kept for experimentation

void set_seeds_const(const int* seeds, int k) {
    CUDA_CHECK(cudaMemcpyToSymbol(const_seeds, seeds, k * sizeof(int)));
}

__global__ void bloom_insert_with_shared_elements_buffer(uint32_t* bit_array, const char* elements, const int* seeds, int elem_size, int num_elems, int m, int k) {
    extern __shared__ char shared_elements[];  // dimensione: blockDim.x * elem_size

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= num_elems) return;

    // Copia la stringa in shared memory
    for (int i = 0; i < elem_size; ++i) {
        shared_elements[tid * elem_size + i] = elements[idx * elem_size + i];
    }
    __syncthreads();  // Assicura che tutti abbiano finito la copia

    // Elabora usando la copia in shared memory
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
    extern __shared__ char shared_elements[];  // blockDim.x * elem_size

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= num_elems) return;

    // Copia la stringa in shared memory
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
