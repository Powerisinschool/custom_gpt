//
// Created by Tolulope Olagunju on 26/05/2026.
//

#ifndef CUSTOMGPT_FEED_FORWARD_HPP
#define CUSTOMGPT_FEED_FORWARD_HPP
#include <vector>

class FeedForward {
public:
    FeedForward(int d_model, int d_ff); // d_ff is usually d_model * 4

    // The forward pass runs on the output of the Attention layer
    std::vector<float> forward(const std::vector<float>& input_sequence, int seq_len) const;

private:
    int d_model;
    int d_ff;

    // The expansion matrix (768 x 3072)
    std::vector<float> W_1;
    std::vector<float> b_1; // Biases

    // The compression matrix (3072 x 768)
    std::vector<float> W_2;
    std::vector<float> b_2; // Biases

    void initialize_weights();
    void load_weights();
};

#endif //CUSTOMGPT_FEED_FORWARD_HPP
