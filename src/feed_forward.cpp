#include "custom_gpt/feed_forward.hpp"

#include <fstream>
#include <iostream>
#include <Accelerate/Accelerate.h>
#include <random>

FeedForward::FeedForward(const int d_model, const int d_ff) : d_model(d_model), d_ff(d_ff) {
    W_1.resize(d_model * d_ff, 0.0f);
    b_1.resize(d_ff, 0.0f);
    W_2.resize(d_ff * d_model, 0.0f);
    b_2.resize(d_model, 0.0f);

    // initialize_weights();
    load_weights();
}

void FeedForward::initialize_weights() {
    const float limit1 = std::sqrt(6.0f / static_cast<float>(d_model + d_ff));
    const float limit2 = std::sqrt(6.0f / static_cast<float>(d_ff + d_model));

    std::random_device rd;
    std::mt19937 generator(42); // Seeded for deterministic debugging TODO: Replace with rd
    std::uniform_real_distribution dist1(-limit1, limit2);
    std::uniform_real_distribution dist2(-limit2, limit1);

    for (float& w : W_1) w = dist1(generator);
    for (float& w : W_2) w = dist2(generator);
    // NB: Biases start at 0.0f, so they remain untouched
}

void FeedForward::load_weights() {
    std::ifstream file1("../bin/w_1.bin", std::ios::binary);
    if (!file1.is_open()) {
        std::cerr << "Error loading weights file " << __FILE__ << " line " << __LINE__ << std::endl;
        return;
    }
    file1.read(reinterpret_cast<char*>(W_1.data()), W_1.size() * sizeof(float));
    file1.close();
    std::ifstream file2("../bin/w_2.bin", std::ios::binary);
    if (!file2.is_open()) {
        std::cerr << "Error loading weights file " << __FILE__ << " line " << __LINE__ << std::endl;
        return;
    }
    file2.read(reinterpret_cast<char*>(W_2.data()), W_2.size() * sizeof(float));
    file2.close();
}

std::vector<float> FeedForward::forward(const std::vector<float> &input_sequence, const int seq_len) const {
    std::vector hidden(seq_len * d_ff, 0.0f);
    std::vector output(seq_len * d_model, 0.0f);

    // hidden = input * W_1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, d_ff, d_model,
                1.0f,
                input_sequence.data(), d_model,
                W_1.data(), d_ff,
                0.0f,
                hidden.data(), d_ff);

    // hidden = hidden + b_1
    for (int i = 0; i < seq_len; i++) {
        // vDSP_vadd adds the bias array to each row of the hidden matrix
        vDSP_vadd(&hidden[i * d_ff], 1, b_1.data(), 1, &hidden[i * d_ff], 1, d_ff);
    }

    // Apply ReLU — max(0, hidden)
    const float zero = 0.0f;
    vDSP_vthres(hidden.data(), 1, &zero, hidden.data(), 1, hidden.size());

    // output = hidden * W_2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, d_model, d_ff,
                1.0f,
                hidden.data(), d_ff,
                W_2.data(), d_model,
                0.0f,
                output.data(), d_model);

    // output = output + b_2
    for (int i = 0; i < seq_len; i++) {
        // vDSP_vadd adds the bias array to each row of the output matrix
        vDSP_vadd(&output[i * d_model], 1, b_2.data(), 1, &output[i * d_model], 1, d_model);
    }

    return output;
}
