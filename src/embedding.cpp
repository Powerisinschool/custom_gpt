#include "custom_gpt/embedding.hpp"
#include <random>
#include <fstream>
#include <iostream>
#include <stdexcept>

// 1. The Heap Allocation
EmbeddingLayer::EmbeddingLayer(int v_size, int e_dim) 
    : vocab_size(v_size), embedding_dim(e_dim) {
    
    // This safely requests a contiguous block of memory on the heap.
    // For a 50,000 x 768 matrix, this instantly allocates ~153 Megabytes.
    weights.resize(vocab_size * embedding_dim);
}

// 2. Training From Scratch (Gaussian Noise)
void EmbeddingLayer::initialize_gaussian() {
    // Neural networks collapse if initialized with zeros. 
    // We use a normal distribution with a mean of 0 and a small standard deviation.
    // GPT-2 uses $\mathcal{N}(0, 0.02)$ for its initial embeddings.
    
    std::mt19937 generator(42); // Seeded for deterministic debugging
    std::normal_distribution distribution(0.0f, 0.02f);

    std::ranges::generate(weights, [&] {return distribution(generator);});
    for (float& weight : weights) {
        weight = distribution(generator);
    }
    std::cout << "Embedding layer initialized with Gaussian noise." << std::endl;
}

// 3. Inference Mode (Loading Pre-trained Weights)
void EmbeddingLayer::load_pretrained(const std::string& filepath) {
    // We open the file in binary mode. Parsing millions of floats from a text file 
    // takes forever. Reading raw bytes directly into memory takes milliseconds.
    std::ifstream file(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open pre-trained weights file.");
    }
    
    // Blast the binary data directly into the vector's memory address
    file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
    file.close();
    std::cout << "Pre-trained weights successfully loaded." << std::endl;
}

// 4. The High-Speed Pointer Extraction
const float* EmbeddingLayer::get_vector(const int token_id) const {
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::out_of_range("Error: Token ID out of bounds.");
    }
    
    // Instead of copying 768 floats, we return a memory address.
    // The exact starting index is always (token_id * embedding_dim).
    return &weights[token_id * embedding_dim];
}
