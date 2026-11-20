//
// Created by Tolulope Olagunju on 25/05/2026.
//

#ifndef CUSTOMGPT_ATTENTION_HPP
#define CUSTOMGPT_ATTENTION_HPP
#include <vector>

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int num_heads);

    // The core forward pass: takes input embeddings and returns attended output
    std::vector<float> forward(const std::vector<float>& input_sequence, int seq_len) const;

private:
    int d_model;       // Typically 768
    int num_heads;     // Typically 12
    int d_k;           // d_model / num_heads (Typically 64)

    // The massive weight matrices (768 x 768 = 589,824 floats each)
    std::vector<float> W_q;
    std::vector<float> W_k;
    std::vector<float> W_v;
    std::vector<float> W_o; // Output projection matrix

    // Initializes the weight matrices with random values (or loads them)
    void initialize_weights();
    void load_weights();
    void check_init_stats() const;
};

#endif //CUSTOMGPT_ATTENTION_HPP
