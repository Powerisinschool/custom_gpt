//
// Created by Tolulope Olagunju on 25/05/2026.
//

#ifndef CUSTOMGPT_TOKENIZER_HPP
#define CUSTOMGPT_TOKENIZER_HPP
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "bpe_node.hpp"

#define INITIAL_VOCAB_SIZE 128

struct TokenPair {
    int left_id;
    int right_id;
    int frequency;

    // TokenPair(int l, int r, int f = 0) : left_id(l), right_id(r), frequency(f) {}
    // TokenPair() = default;

    bool operator<(const TokenPair& other) const {
        if (frequency != other.frequency) return frequency < other.frequency;
        if (left_id != other.left_id) return left_id < other.left_id;
        return right_id < other.right_id;
    }
};

class Tokenizer {
public:
    Tokenizer();

    void train(const std::string& filepath, int target_vocab_size);
    void print_vocab() const;
    std::vector<int> encode(const std::string& text);
    [[nodiscard]] std::string decode(const std::vector<int>& ids, bool shouldSeparate=false) const;
    int get_vocab_size() const;
private:
    std::unordered_map<std::string, int> word_counts;
    std::vector<Node*> active_words;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> reverse_vocab;
    int vocab_size;
    // int vocab_size_limit;
    std::unordered_map<uint64_t, TokenPair> paired_frequencies;
    std::priority_queue<TokenPair> maxHeap;
    std::vector<std::pair<int, int>> merge_rules;

    void process_chunk(const std::string& chunk);
    void build_word_lists();
    void tally_pairs();
    void apply_merge(int id_left, int id_right, int new_id);
};

#endif //CUSTOMGPT_TOKENIZER_HPP
