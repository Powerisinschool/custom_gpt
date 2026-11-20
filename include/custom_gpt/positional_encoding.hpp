//
// Created by Tolulope Olagunju on 25/05/2026.
//

#ifndef CUSTOMGPT_POSITIONAL_ENCODING_HPP
#define CUSTOMGPT_POSITIONAL_ENCODING_HPP
#include <vector>

class PositionalEncoding {
public:
    // Initializes the wave matrix for a given context length and dimension size
    PositionalEncoding(int max_seq_len, int e_dim);

    // Returns the exact memory address for the wave frequencies at a specific position
    const float* get_vector(int position) const;

private:
    int max_sequence_length;
    int embedding_dim;

    // Contiguous 1D array to prevent CPU cache misses
    std::vector<float> pe_matrix;
};

#endif //CUSTOMGPT_POSITIONAL_ENCODING_HPP
