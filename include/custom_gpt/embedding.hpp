//
// Created by Tolulope Olagunju on 25/05/2026.
//

#ifndef CUSTOMGPT_EMBEDDING_HPP
#define CUSTOMGPT_EMBEDDING_HPP
#include <vector>

class EmbeddingLayer {
public:
    // Constructor to allocate the massive float array
    EmbeddingLayer(int v_size, int e_dim);

    void initialize_gaussian();

    void load_pretrained(const std::string &filepath);

    // Returns a pointer to the start of the exact 768-dim vector for a token
    [[nodiscard]] const float* get_vector(int token_id) const;

private:
    int vocab_size;
    int embedding_dim;

    // The flat 1D array masquerading as a 2D matrix
    std::vector<float> weights;
};

#endif //CUSTOMGPT_EMBEDDING_HPP
