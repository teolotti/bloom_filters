#include "utils/BloomFilters.h"
#include "utils/cuda_bloom.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>



// Genera dati random per test
std::vector<std::string> generate_random_data(int num_elements, int max_len = 20) {
    std::vector<std::string> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<char> char_dist('a', 'z');


    for (int i = 0; i < num_elements; ++i) {
        int len = 5 + (gen() % max_len);
        std::string s;
        for (int j = 0; j < len; ++j) s += char_dist(gen);
        data.push_back(s);
    }
    return data;
}

int main() {
    // ========== Configurazione ==========
    const int m = 1 << 24;          // 16M bit (~2MB)
    const int k = 5;                // 3 funzioni hash
    const int num_elements = 1 << 20; // 1M elementi
    const int block_size = 256;     // Threads per blocco
    const int grid_size = 1 << 10; // Blocchi per griglia (1024 blocchi)
    const int h_seeds[5] = { 0x12345678, 0x23456789, 0x34567890, 0x45678901, 0x56789012 };

    // Genera dati di test
    std::vector<std::string> elements = generate_random_data(num_elements);
    std::vector<std::string> query_elements = generate_random_data(num_elements);

    // ========== Benchmark CPU ==========
    BloomFilters bf_cpu(m, k);

    // Inserimento CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (const auto& elem : elements) bf_cpu.insert(elem);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_insert_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // Query CPU
    start_cpu = std::chrono::high_resolution_clock::now();
    for (const auto& elem : query_elements) bf_cpu.query(elem);
    end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_query_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // ========== Setup GPU ==========
    // Converti elementi in formato flattened (array di char concatenati)
    std::vector<char> flat_elements, flat_queries;
    int max_len = 0;
    for (const auto& s : elements) {
        flat_elements.insert(flat_elements.end(), s.begin(), s.end());
        flat_elements.push_back('\0');
        max_len = std::max(max_len, (int)s.size() + 1);
    }
    for (const auto& s : query_elements) {
        flat_queries.insert(flat_queries.end(), s.begin(), s.end());
        flat_queries.push_back('\0');
    }

    // Alloca memoria GPU
    uint32_t *d_bit_array;
    char *d_elements, *d_queries;
    bool *d_results;
    int *d_seeds;
    CUDA_CHECK(cudaMalloc(&d_bit_array, ((m + 31) / 32) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_elements, flat_elements.size()));
    CUDA_CHECK(cudaMalloc(&d_queries, flat_queries.size()));
    CUDA_CHECK(cudaMalloc(&d_results, num_elements * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_seeds, 5 * sizeof(int)));

    // Copia dati su GPU
    CUDA_CHECK(cudaMemcpy(d_elements, flat_elements.data(), flat_elements.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, flat_queries.data(), flat_queries.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, 5 * sizeof(int), cudaMemcpyHostToDevice));

    // ========== Benchmark GPU ==========
    // ---- Versione Baseline ----
    CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
    auto start_gpu = std::chrono::high_resolution_clock::now();
    bloom_insert_kernel<<<grid_size, block_size>>>(d_bit_array, d_elements, d_seeds, max_len, num_elements, m, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_baseline_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // ---- Versione Constant Memory ----
    set_seeds_const(h_seeds, 5);

    CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
    start_gpu = std::chrono::high_resolution_clock::now();
    bloom_insert_constant<<<grid_size, block_size>>>(d_bit_array, d_elements, max_len, num_elements, m, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_constant_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // ---- Versione Shared Memory ----
    size_t shared_mem_size = ((m + 31) / 32) * sizeof(uint32_t);
    CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
    start_gpu = std::chrono::high_resolution_clock::now();
    bloom_insert_shared<<<grid_size, block_size, shared_mem_size>>>(d_bit_array, d_elements, d_seeds, max_len, num_elements, m, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_shared_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // ========== Risultati ==========
    std::cout << "=== Benchmark Bloom Filter ===\n";
    std::cout << "Configurazione: m=" << m << " bit, k=" << k << ", elementi=" << num_elements << "\n\n";

    std::cout << "CPU (Sequenziale):\n";
    std::cout << "  Inserimento: " << cpu_insert_time << " s\n";
    std::cout << "  Query: " << cpu_query_time << " s\n\n";

    std::cout << "GPU (Baseline):\n";
    std::cout << "  Inserimento: " << gpu_baseline_insert_time << " s (Speedup: " << cpu_insert_time / gpu_baseline_insert_time << "x)\n\n";

    std::cout << "GPU (Constant Memory):\n";
    std::cout << "  Inserimento: " << gpu_constant_insert_time << " s (Speedup: " << cpu_insert_time / gpu_constant_insert_time << "x)\n\n";

    std::cout << "GPU (Shared Memory):\n";
    std::cout << "  Inserimento: " << gpu_shared_insert_time << " s (Speedup: " << cpu_insert_time / gpu_shared_insert_time << "x)\n";

    // ========== Pulizia ==========
    CUDA_CHECK(cudaFree(d_bit_array));
    CUDA_CHECK(cudaFree(d_elements));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}