#include "custom_gpt/positional_encoding.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

PositionalEncoding::PositionalEncoding(const int max_seq_len, const int e_dim)
    : max_sequence_length(max_seq_len), embedding_dim(e_dim) {

    // Allocate memory and initialize to 0.0f
    pe_matrix.resize(max_sequence_length * embedding_dim, 0.0f);

    // Pre-calculate the exact wave value for every dimension at every position
    for (int pos = 0; pos < max_sequence_length; ++pos) {
        
        // We step by 2 because we process the Even (Sine) and Odd (Cosine) pairs together
        for (int i = 0; i < embedding_dim; i += 2) {
            
            // The scaling denominator: 10000^(2i / d_model)
            const float denominator = std::pow(10000.0f, static_cast<float>(i) / embedding_dim);
            
            // Calculate our starting index in the 1D array
            const int base_index = (pos * embedding_dim);

            // Even dimensions get the Sine wave
            pe_matrix[base_index + i] = std::sin(pos / denominator);

            // Odd dimensions get the Cosine wave (with a safety check for odd-numbered dimension limits)
            if (i + 1 < embedding_dim) {
                pe_matrix[base_index + i + 1] = std::cos(pos / denominator);
            }
        }
    }
    std::cout << "Positional Encoding matrix generated successfully." << std::endl;
}

const float* PositionalEncoding::get_vector(const int position) const {
    if (position < 0 || position >= max_sequence_length) {
        throw std::out_of_range("Error: Sequence position out of bounds.");
    }
    return &pe_matrix[position * embedding_dim];
}
