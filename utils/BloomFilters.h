//
// Created by matteo on 09/07/25.
//

#ifndef BLOOM_FILTERS_BLOOMFILTERS_H
#define BLOOM_FILTERS_BLOOMFILTERS_H


#include <cstdint>
#include <vector>
#include <string>



class BloomFilters {
private:
    std::vector<uint32_t> bitArray;
    int k; // Number of hash functions
    int m; // Size of the bit array
    std::vector<int> seeds; // Seeds for hash functions

    size_t hash(const std::string &element, int seed) const {  // FNV-1a hash function
        size_t h = 0;
        for (char c : element) {
            h = (h * 31 + c) % m; // Simple hash function
        }
        return (h + seed) % m; // Use seed to create different hash functions
    }
public:
    BloomFilters(int size, int numHashFunctions, std::vector<int> seedArray) : m(size), k(numHashFunctions), seeds(std::move(seedArray)) {
        bitArray.resize((m + 31) / 32, 0); // Initialize bit array
    }

    void insert(const std::string &element) {
        for (int i = 0; i < k; ++i) {
            size_t index = hash(element, this->seeds[i]);
            bitArray[index / 32] |= (1u << (index % 32));
        }
    }

    bool query(const std::string &element) const {
        for (int i = 0; i < k; ++i) {
            size_t index = hash(element, this->seeds[i]);
            if (!(bitArray[index / 32] & (1u << (index % 32)))) {
                return false;
            }
        }
        return true;
    }
};


#endif //BLOOM_FILTERS_BLOOMFILTERS_H
