//
// Created by Tolulope Olagunju on 26/05/2026.
//

#ifndef CUSTOMGPT_TRANSFORMER_BLOCK_HPP
#define CUSTOMGPT_TRANSFORMER_BLOCK_HPP
#include <vector>

#include "attention.hpp"
#include "feed_forward.hpp"

class TransformerBlock {
public:
    TransformerBlock(int d_model, int num_heads, int d_ff);

    // The main routing function
    std::vector<float> forward(const std::vector<float>& input_sequence, int seq_len) const;

private:
    int d_model;

    MultiHeadAttention attention;
    FeedForward feed_forward;

    // Layer Normalization parameters (gamma = scale, beta = shift)
    std::vector<float> ln1_gamma;
    std::vector<float> ln1_beta;
    std::vector<float> ln2_gamma;
    std::vector<float> ln2_beta;

    // Helper function for Layer Norm
    std::vector<float> layer_norm(const std::vector<float>& x,
                                  const std::vector<float>& gamma,
                                  const std::vector<float>& beta,
                                  int seq_len) const;
};

#endif //CUSTOMGPT_TRANSFORMER_BLOCK_HPP
