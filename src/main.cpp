#include <fstream>

#include "custom_gpt/embedding.hpp"
#include "custom_gpt/positional_encoding.hpp"
#include "custom_gpt/transformer_block.hpp"
#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>

#include "custom_gpt/tokenizer.hpp"

int main() {
    Tokenizer tokenizer;
    tokenizer.train("../data/shakespeare.txt", 15000);
    // 1. System Parameters
    const int VOCAB_SIZE = tokenizer.get_vocab_size();
    constexpr int D_MODEL = 128; // 768
    constexpr int MAX_SEQ_LEN = 1024;
    constexpr int NUM_HEADS = 12;
    constexpr int D_FF = 512; // 3072
    constexpr int MAX_NEW_TOKENS = 20;

    std::cout << "--- ALLOCATING AI MEMORY ---" << std::endl;
    // (Assuming tokenizer is already trained and loaded in a real scenario)
    EmbeddingLayer embeddings(VOCAB_SIZE, D_MODEL);
    // embeddings.initialize_gaussian();
    embeddings.load_pretrained("../bin/embeddings.bin");

    const PositionalEncoding pos_encoding(MAX_SEQ_LEN, D_MODEL);
    const TransformerBlock block(D_MODEL, NUM_HEADS, D_FF);

    std::vector<float> lm_head(VOCAB_SIZE * D_MODEL);
    {
        std::ifstream f("../bin/lm_head.bin", std::ios::binary);
        if (!f) { std::cerr << "Failed to open lm_head.bin\n"; return 1; }
        f.read(reinterpret_cast<char*>(lm_head.data()),
               lm_head.size() * sizeof(float));
    }
    std::cout << "LM Head loaded." << std::endl;

    // 2. The Input Text
    // Let's pretend the tokenizer output this ID for "the"
    constexpr std::string start_text = "First Citizen";
    std::vector token_ids = tokenizer.encode(start_text);
    std::cout << "Starting text: '" << start_text << "'" << std::endl;
    std::cout << "LM Token IDs:" << std::endl;
    for (size_t i = 0; i < token_ids.size(); i++)
        std::cout << token_ids[i] << " ";
    std::cout << std::endl;

    for (int step = 0; step < MAX_NEW_TOKENS; step++) {
        const int seq_len = static_cast<int>(token_ids.size());

        // 3. Mapping: Integers -> Geometry + Time
        std::vector input_sequence(seq_len * D_MODEL, 0.0f);

        for (int pos = 0; pos < seq_len; pos++) {
            const int id = token_ids[pos];
            const float* word_vec = embeddings.get_vector(id);
            const float* time_vec = pos_encoding.get_vector(pos);

            // Add the Word Geometry and the Positional Wave together
            for (int i = 0; i < D_MODEL; i++) {
                input_sequence[pos * D_MODEL + i] = word_vec[i] + time_vec[i];
            }
        }

        std::cout << "--- EXECUTING FORWARD PASS ---" << std::endl;
        // 4. The Brain: Pushing the data through the Transformer Block
        const std::vector<float> output_sequence = block.forward(input_sequence, seq_len);

        // The last token's output vector is what we predict the *next* token from
        const float* last_token_vec = &output_sequence[(seq_len - 1) * D_MODEL];

        // Produces a score (logit) for every word in the vocabulary
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    VOCAB_SIZE, D_MODEL,
                    1.0f, lm_head.data(), D_MODEL,
                          last_token_vec,  1,
                    0.0f, logits.data(),   1);

        const int next_id = static_cast<int>(
            std::ranges::max_element(logits) - logits.begin()
        );

        std::cout << "Step " << step + 1 << ": predicted token ID " << next_id << std::endl;

        // Append and loop
        token_ids.push_back(next_id);
    }

    // 5. Verification
    std::cout << "\nSuccess! No segmentation faults." << std::endl;
    // std::cout << "Input Sequence Size: " << input_sequence.size() << " floats." << std::endl;
    // std::cout << "Output Sequence Size: " << output_sequence.size() << " floats." << std::endl;

    std::cout << "\nOutput Vector for Token [0] (First 5 dimensions): [ ";
    // for (int i = 0; i < 5; i++) {
    //     std::cout << output_sequence[i] << " ";
    // }
    // std::cout << "... ]" << std::endl;
    for (int id : token_ids) std::cout << id << " ";
    std::cout << "]" << std::endl;

    std::cout << "\nText:" << std::endl;
    std::cout << tokenizer.decode(token_ids, true) << std::endl;

    // std::cout << tokenizer << std::endl;

    return 0;
}
