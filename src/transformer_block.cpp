#include "custom_gpt/transformer_block.hpp"
#include <Accelerate/Accelerate.h>
#include <cmath>

TransformerBlock::TransformerBlock(const int d_model, const int num_heads, const int d_ff)
    : d_model(d_model), attention(d_model, num_heads), feed_forward(d_model, d_ff) {
    
    // Gamma starts at 1.0 (no scaling), Beta starts at 0.0 (no shift)
    ln1_gamma.resize(d_model, 1.0f);
    ln1_beta.resize(d_model, 0.0f);
    ln2_gamma.resize(d_model, 1.0f);
    ln2_beta.resize(d_model, 0.0f);
}

std::vector<float> TransformerBlock::layer_norm(const std::vector<float>& x, 
                                                const std::vector<float>& gamma, 
                                                const std::vector<float>& beta,
                                                const int seq_len) const {
    std::vector<float> out(x.size());
    constexpr float epsilon = 1e-5f; // Prevent division by zero

    for (int i = 0; i < seq_len; i++) {
        const float* token_vec = &x[i * d_model];
        float* out_vec = &out[i * d_model];

        // 1. Calculate Mean
        float mean;
        vDSP_meanv(token_vec, 1, &mean, d_model);

        // 2. Calculate Variance
        float variance = 0.0f;
        for (int j = 0; j < d_model; j++) {
            const float diff = token_vec[j] - mean;
            variance += diff * diff;
        }
        variance /= d_model;

        // 3. Normalize, Scale (gamma), and Shift (beta)
        const float std_dev = std::sqrt(variance + epsilon);
        for (int j = 0; j < d_model; j++) {
            out_vec[j] = ((token_vec[j] - mean) / std_dev) * gamma[j] + beta[j];
        }
    }
    return out;
}

std::vector<float> TransformerBlock::forward(const std::vector<float>& input_sequence, const int seq_len) const {
    // --- PART 1: MULTI-HEAD ATTENTION ---
    // Pre-Layer Normalization
    const std::vector<float> norm1 = layer_norm(input_sequence, ln1_gamma, ln1_beta, seq_len);
    
    // Attention Pass
    const std::vector<float> attn_out = attention.forward(norm1, seq_len);
    
    // Residual Connection 1 (Input + Attention Output)
    std::vector<float> res1(input_sequence.size());
    vDSP_vadd(input_sequence.data(), 1, attn_out.data(), 1, res1.data(), 1, input_sequence.size());

    // --- PART 2: FEED-FORWARD NETWORK ---
    // Pre-Layer Normalization
    const std::vector<float> norm2 = layer_norm(res1, ln2_gamma, ln2_beta, seq_len);
    
    // Feed-Forward Pass
    const std::vector<float> ff_out = feed_forward.forward(norm2, seq_len);
    
    // Residual Connection 2 (Res1 + FF Output)
    std::vector<float> final_out(input_sequence.size());
    vDSP_vadd(res1.data(), 1, ff_out.data(), 1, final_out.data(), 1, input_sequence.size());

    return final_out;
}
