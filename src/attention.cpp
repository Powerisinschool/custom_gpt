#include "custom_gpt/attention.hpp"

#include <fstream>
#include <Accelerate/Accelerate.h>
#include <iostream>
#include <random>

MultiHeadAttention::MultiHeadAttention(const int d_model, const int num_heads)
    : d_model(d_model), num_heads(num_heads) {
    d_k = d_model / num_heads;

    // Allocate continuous memory for our weight matrices
    const int matrix_size = d_model * d_model;
    W_q.resize(matrix_size, 0.0f);
    W_k.resize(matrix_size, 0.0f);
    W_v.resize(matrix_size, 0.0f);
    W_o.resize(matrix_size, 0.0f);

    // initialize_weights();
    load_weights();
    check_init_stats();
}

std::vector<float> MultiHeadAttention::forward(const std::vector<float>& input_sequence, const int seq_len) const {
    // 1. Generate Q, K, V matrices by multiplying the input by the weights
    // (In reality, this is where you'd call a highly optimized math library)
    // std::vector<float> Q(seq_len * d_model, 0.0f);
    // std::vector<float> K(seq_len * d_model, 0.0f);
    // std::vector<float> V(seq_len * d_model, 0.0f);
    auto gemm = [](const float* A, const float* B, float* C,
                    const int M, const int N, const int K) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f, A, K,
                        B, N,
                    0.0f, C, N);
    };

    std::vector<float> Q(seq_len * d_model), K(seq_len * d_model), V(seq_len * d_model);

    // ... [Matrix Multiplication Logic: Q = input * W_q] ...
    gemm(input_sequence.data(), W_q.data(), Q.data(), seq_len, d_model, d_model);
    gemm(input_sequence.data(), W_k.data(), K.data(), seq_len, d_model, d_model);
    gemm(input_sequence.data(), W_v.data(), V.data(), seq_len, d_model, d_model);

    std::vector concat_output(seq_len * d_model, 0.0f);
    // std::vector<float> scores(num_heads * seq_len * seq_len, 0.0f);

    for (int h = 0; h < num_heads; h++) {
        const int col = h * d_k;
        // for (int i = 0; i < seq_len; i++) {       // For every query word
        //     for (int j = 0; j < seq_len; j++) {   // Against every key word
        //         float dot_product = 0.0f;
        //         for (int k = 0; k < d_k; k++) {
        //             // We extract the dot product for the current head
        //             dot_product += Q[i * d_model + col + k] * K[j * d_model + col + k];
        //         }
        //         // Scale it down by the square root of d_k to prevent vanishing gradients
        //         scores[h * seq_len * seq_len + i * seq_len + j] = dot_product / std::sqrt(static_cast<float>(d_k));
        //     }
        // }
        std::vector<float> Qh(seq_len * d_k), Kh(seq_len * d_k), Vh(seq_len * d_k);
        for (int i = 0; i < seq_len; i++) {
            std::copy_n(&Q[i * d_model + col], d_k, &Qh[i * d_k]);
            std::copy_n(&K[i * d_model + col], d_k, &Kh[i * d_k]);
            std::copy_n(&V[i * d_model + col], d_k, &Vh[i * d_k]);
        }

        // 2. Calculate Attention Scores (Q * K^T)
        std::vector scores(seq_len * seq_len, 0.0f);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, d_k,
                    1.0f / std::sqrt(static_cast<float>(d_k)),
                    Qh.data(), d_k,
                    Kh.data(), d_k,
                    0.0f, scores.data(), seq_len);

        for (int i = 0; i < seq_len; i++)
            for (int j = i + 1; j < seq_len; j++)
                scores[i * seq_len + j] = -std::numeric_limits<float>::infinity();

        // 3. Apply Softmax to turn the scores into probabilities (0.0 to 1.0)
        // ... [Softmax Logic: Exponentiate and normalize the rows] ...
        for (int i = 0; i < seq_len; i++) {
            float *row = &scores[i * seq_len];
            float max_val;
            vDSP_maxv(row, 1, &max_val, seq_len); // find max
            float neg_max = -max_val;
            vDSP_vsadd(row, 1, &neg_max, row, 1, seq_len); // subtract max
            vvexpf(row, row, &seq_len); // exp() (vectorized)
            float sum;
            vDSP_sve(row, 1, &sum, seq_len); // sum
            vDSP_vsdiv(row, 1, &sum, row, 1, seq_len); // divide
        }

        // 4. Multiply the resulting probabilities by the Value matrix (V)
        std::vector head_output(seq_len * d_k, 0.0f);
        // ... [Matrix Multiplication Logic: Output = Softmax(Scores) * V] ...
        gemm(scores.data(), Vh.data(), head_output.data(), seq_len, d_k, seq_len);
        for (int i = 0; i < seq_len; i++)
            std::copy_n(&head_output[i * d_k], d_k, &concat_output[i * d_model + col]);
    }

    // 5. Final output projection
    std::vector<float> output(seq_len * d_model);
    // ... [Matrix Multiplication Logic: Final = Output * W_o] ...
    gemm(concat_output.data(), W_o.data(), output.data(), seq_len, d_model, d_model);

    return output;
}

void MultiHeadAttention::initialize_weights() {
    // Xavier uniform: limit = sqrt(3 / d_model) when fan_in == fan_out
    const float limit = std::sqrt(3.0f / static_cast<float>(d_model));

    std::random_device rd;
    std::mt19937 generator(42); // Seeded for deterministic debugging TODO: Replace with rd
    std::uniform_real_distribution distribution(-limit, limit);

    auto fill = [&](std::vector<float>& w) {
        std::ranges::generate(w, [&] {return distribution(generator);});
    };

    fill(W_q);
    fill(W_k);
    fill(W_v);
    fill(W_o);
}

void MultiHeadAttention::load_weights() {
    std::ifstream file1("../bin/w_q.bin", std::ios::binary);
    if (!file1.is_open()) {
        std::cerr << "Error loading weights file " << __FILE__ << " line " << __LINE__ << std::endl;
        return;
    }
    file1.read(reinterpret_cast<char*>(W_q.data()), W_q.size() * sizeof(float));
    file1.close();
    std::ifstream file2("../bin/w_k.bin", std::ios::binary);
    if (!file2.is_open()) {
        std::cerr << "Error loading weights file " << __FILE__ << " line " << __LINE__ << std::endl;
        return;
    }
    file2.read(reinterpret_cast<char*>(W_k.data()), W_k.size() * sizeof(float));
    file2.close();
    std::ifstream file3("../bin/w_v.bin", std::ios::binary);
    if (!file3.is_open()) {
        std::cerr << "Error loading weights file " << __FILE__ << " line " << __LINE__ << std::endl;
        return;
    }
    file3.read(reinterpret_cast<char*>(W_v.data()), W_v.size() * sizeof(float));
    file3.close();
    std::ifstream file4("../bin/w_o.bin", std::ios::binary);
    if (!file4.is_open()) {
        std::cerr << "Error loading weights file " << __FILE__ << " line " << __LINE__ << std::endl;
        return;
    }
    file4.read(reinterpret_cast<char*>(W_o.data()), W_o.size() * sizeof(float));
    file4.close();
}

void MultiHeadAttention::check_init_stats() const {
    auto stats = [](const std::vector<float>& w, const std::string& name) {
        const float mean = std::accumulate(w.begin(), w.end(), 0.0f) / w.size();
        const float sq_sum = std::inner_product(w.begin(), w.end(), w.begin(), 0.0f);
        const float std_dev = std::sqrt(sq_sum / w.size() - mean * mean);
        std::cout << name << ": mean=" << mean << " std=" << std_dev << "\n";
    };

    const float expected_std = std::sqrt(3.0f / static_cast<float>(d_model)) / std::sqrt(3.0f);
    std::cout << "Expected std ≈ " << expected_std << "\n";
    stats(W_q, "W_q");
    stats(W_k, "W_k");
    stats(W_v, "W_v");
    stats(W_o, "W_o");
}
