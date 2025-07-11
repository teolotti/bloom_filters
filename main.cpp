#include "utils/BloomFilters.h"
#include "utils/cuda_bloom.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>


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

void write_csv_row(std::ofstream& file, int block_size, int m, int num_elements,
                   double cpu_insert, double cpu_query,
                   double gpu_insert, double gpu_query) {
    file << block_size << "," << m << "," << num_elements << ","
         << std::setprecision(6) << cpu_insert << "," << cpu_query << ","
         << gpu_insert << "," << gpu_query << "\n";
}

int main() {
    // Configurazione parametri
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    std::vector<int> m_values, num_elements_values;
    for (int i = 14; i <= 24; i += 2) m_values.push_back(1 << i);
    for (int i = 10; i <= 20; i += 2) num_elements_values.push_back(1 << i);

    // File CSV per ogni versione
    std::ofstream csv_base("benchmark_base.csv");
    std::ofstream csv_const("benchmark_constant.csv");
    std::ofstream csv_shared("benchmark_shared.csv");
    std::ofstream csv_shared_buf("benchmark_shared_buffer.csv");
    std::ofstream csv_hybrid("benchmark_hybrid.csv");

    // Header CSV
    std::string header = "block_size,m,num_elements,cpu_insert,cpu_query,gpu_insert,gpu_query\n";
    csv_base << header;
    csv_const << header;
    csv_shared << header;
    csv_shared_buf << header;
    csv_hybrid << header;
    int k = NUM_SEEDS;
    for (int block_size : block_sizes) {
        for (int m : m_values) {
            for (int num_elements : num_elements_values) {
                // Genera dati random
                std::vector<std::string> elements = generate_random_data(num_elements);
                std::vector<std::string> query_elements = generate_random_data(num_elements);


                std::vector<int> h_seeds = {0x12345678, 0x23456789, 0x34567890, 0x45678901, 0x56789012};
                if (k == 7) {
                    h_seeds.push_back(0x67890123);
                    h_seeds.push_back(0x78901234);
                }
                int h_seeds_vec[NUM_SEEDS];
                for (int i = 0; i < k; ++i) {
                    h_seeds_vec[i] = h_seeds[i];
                }
                int max_len = 0;
                for (const auto& s : elements) max_len = std::max(max_len, (int)s.size() + 1);

                // CPU benchmark
                BloomFilters bf_cpu(m, k, h_seeds);
                auto start_cpu = std::chrono::high_resolution_clock::now();
                for (const auto& elem : elements) bf_cpu.insert(elem);
                auto end_cpu = std::chrono::high_resolution_clock::now();
                double cpu_insert_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

                start_cpu = std::chrono::high_resolution_clock::now();
                for (const auto& elem : query_elements) bf_cpu.query(elem);
                end_cpu = std::chrono::high_resolution_clock::now();
                double cpu_query_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

                // Prepara dati per GPU
                std::vector<char> flat_elements(num_elements * max_len, '\0');
                std::vector<char> flat_queries(num_elements * max_len, '\0');
                for (int i = 0; i < num_elements; ++i) {
                    strncpy(&flat_elements[i * max_len], elements[i].c_str(), max_len - 1);
                    strncpy(&flat_queries[i * max_len], query_elements[i].c_str(), max_len - 1);
                }
                uint32_t *d_bit_array;
                char *d_elements, *d_queries;
                bool *d_results;
                int *d_seeds;
                CUDA_CHECK(cudaMalloc(&d_bit_array, ((m + 31) / 32) * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&d_elements, num_elements * max_len));
                CUDA_CHECK(cudaMalloc(&d_queries, num_elements * max_len));
                CUDA_CHECK(cudaMalloc(&d_results, num_elements * sizeof(bool)));
                CUDA_CHECK(cudaMalloc(&d_seeds, k * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(d_elements, flat_elements.data(), num_elements * max_len, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_queries, flat_queries.data(), num_elements * max_len, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds_vec, k * sizeof(int), cudaMemcpyHostToDevice));
                int grid_size = (num_elements + block_size - 1) / block_size;

                // --- BASELINE ---
                CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
                auto start_gpu = std::chrono::high_resolution_clock::now();
                bloom_insert_kernel<<<grid_size, block_size>>>(d_bit_array, d_elements, d_seeds, max_len, num_elements, m, k);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                auto end_gpu = std::chrono::high_resolution_clock::now();
                double gpu_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                CUDA_CHECK(cudaMemset(d_results, 0, num_elements * sizeof(bool)));
                start_gpu = std::chrono::high_resolution_clock::now();
                bloom_query_kernel<<<grid_size, block_size>>>(d_bit_array, d_queries, d_seeds, max_len, num_elements, m, k, d_results);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                end_gpu = std::chrono::high_resolution_clock::now();
                double gpu_query_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                write_csv_row(csv_base, block_size, m, num_elements, cpu_insert_time, cpu_query_time, gpu_insert_time, gpu_query_time);

                // --- CONSTANT ---
                set_seeds_const(h_seeds_vec, k);
                CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
                start_gpu = std::chrono::high_resolution_clock::now();
                bloom_insert_constant<<<grid_size, block_size>>>(d_bit_array, d_elements, max_len, num_elements, m, k);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                end_gpu = std::chrono::high_resolution_clock::now();
                double gpu_const_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                CUDA_CHECK(cudaMemset(d_results, 0, num_elements * sizeof(bool)));
                start_gpu = std::chrono::high_resolution_clock::now();
                bloom_query_constant<<<grid_size, block_size>>>(d_bit_array, d_queries, max_len, num_elements, m, k, d_results);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                end_gpu = std::chrono::high_resolution_clock::now();
                double gpu_const_query_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                write_csv_row(csv_const, block_size, m, num_elements, cpu_insert_time, cpu_query_time, gpu_const_insert_time, gpu_const_query_time);

                // --- SHARED ELEMENT BUFFER ---
                size_t shared_mem_size = block_size * max_len * sizeof(char);
                CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
                start_gpu = std::chrono::high_resolution_clock::now();
                bloom_insert_with_shared_elements_buffer<<<grid_size, block_size, shared_mem_size>>>(d_bit_array, d_elements, d_seeds, max_len, num_elements, m, k);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                end_gpu = std::chrono::high_resolution_clock::now();
                double gpu_shared_buf_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                CUDA_CHECK(cudaMemset(d_results, 0, num_elements * sizeof(bool)));
                start_gpu = std::chrono::high_resolution_clock::now();
                bloom_query_with_shared_elements_buffer<<<grid_size, block_size, shared_mem_size>>>(d_bit_array, d_queries, d_seeds, max_len, num_elements, m, k, d_results);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                end_gpu = std::chrono::high_resolution_clock::now();
                double gpu_shared_buf_query_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                write_csv_row(csv_shared_buf, block_size, m, num_elements, cpu_insert_time, cpu_query_time, gpu_shared_buf_insert_time, gpu_shared_buf_query_time);

                // --- SHARED & HYBRID solo se m <= 48KB ---
                if (m <= (48 * 1024 * 8)) {
                    // SHARED
                    CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
                    start_gpu = std::chrono::high_resolution_clock::now();
                    bloom_insert_shared<<<grid_size, block_size>>>(d_bit_array, d_elements, d_seeds, max_len, num_elements, m, k);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    end_gpu = std::chrono::high_resolution_clock::now();
                    double gpu_shared_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                    CUDA_CHECK(cudaMemset(d_results, 0, num_elements * sizeof(bool)));
                    start_gpu = std::chrono::high_resolution_clock::now();
                    bloom_query_shared<<<grid_size, block_size>>>(d_bit_array, d_queries, d_seeds, max_len, num_elements, m, k, d_results);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    end_gpu = std::chrono::high_resolution_clock::now();
                    double gpu_shared_query_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                    write_csv_row(csv_shared, block_size, m, num_elements, cpu_insert_time, cpu_query_time, gpu_shared_insert_time, gpu_shared_query_time);

                    // HYBRID
                    CUDA_CHECK(cudaMemset(d_bit_array, 0, ((m + 31) / 32) * sizeof(uint32_t)));
                    start_gpu = std::chrono::high_resolution_clock::now();
                    bloom_insert_hybrid<<<grid_size, block_size>>>(d_bit_array, d_elements, max_len, num_elements, m, k);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    end_gpu = std::chrono::high_resolution_clock::now();
                    double gpu_hybrid_insert_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                    CUDA_CHECK(cudaMemset(d_results, 0, num_elements * sizeof(bool)));
                    start_gpu = std::chrono::high_resolution_clock::now();
                    bloom_query_hybrid<<<grid_size, block_size>>>(d_bit_array, d_queries, max_len, num_elements, m, k, d_results);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    end_gpu = std::chrono::high_resolution_clock::now();
                    double gpu_hybrid_query_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

                    write_csv_row(csv_hybrid, block_size, m, num_elements, cpu_insert_time, cpu_query_time, gpu_hybrid_insert_time, gpu_hybrid_query_time);
                }

                // Pulizia
                CUDA_CHECK(cudaFree(d_bit_array));
                CUDA_CHECK(cudaFree(d_elements));
                CUDA_CHECK(cudaFree(d_queries));
                CUDA_CHECK(cudaFree(d_results));
                CUDA_CHECK(cudaFree(d_seeds));
            }
        }
    }

    // Chiudi file
    csv_base.close();
    csv_const.close();
    csv_shared.close();
    csv_shared_buf.close();
    csv_hybrid.close();

    return 0;
}