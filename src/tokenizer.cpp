#include "custom_gpt/tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <ranges>

Tokenizer::Tokenizer() {
    // Pre-populate the base 256 byte vocabulary
    for (int i = 0; i < INITIAL_VOCAB_SIZE; i++) {
        // Create a string from the single raw byte
        std::string byte_str(1, static_cast<char>(i));

        vocab[byte_str] = i;
        reverse_vocab.push_back(byte_str);
    }
    vocab_size = INITIAL_VOCAB_SIZE;
}

void Tokenizer::train(const std::string &filepath, const int target_vocab_size) {
    if (filepath.empty()) {
        std::cerr << "No file path provided" << std::endl;
        return;
    }
    if (target_vocab_size < INITIAL_VOCAB_SIZE) {
        std::cerr << "Vocab size is too small (" << target_vocab_size << " < " << INITIAL_VOCAB_SIZE << ")" <<
                std::endl;
        return;
    }
    std::ifstream inputFile(filepath);

    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open the file!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        line += '\n';
        process_chunk(line);
    }

    inputFile.close();

    build_word_lists();
    tally_pairs();
    while (vocab_size < target_vocab_size && !maxHeap.empty()) {
        const TokenPair pair = maxHeap.top();
        apply_merge(pair.left_id, pair.right_id, vocab_size);
        maxHeap.pop();
        vocab_size++;
    }
    // vocab_size_limit = target_vocab_size;
}

void Tokenizer::process_chunk(const std::string& chunk) {
    std::string current_word;
    for (const char c : chunk) {
        if (const auto uc = static_cast<unsigned char>(c); std::isalpha(uc) || std::isdigit(uc)) {
            current_word += c;
        }
        // else if (std::isspace(uc) || std::ispunct(uc)) {
        //     if (!current_word.empty()) {
        //         word_counts[current_word]++;
        //         current_word = "";
        //     }
        //     word_counts[std::string(1, c)]++;
        // }
        else {
            if (!current_word.empty()) {
                word_counts[current_word]++;
                current_word = "";
            }
            word_counts[std::string(1, c)]++;
        }
    }
    if (!current_word.empty()) {
        word_counts[current_word]++;
    }
}

void Tokenizer::build_word_lists() {
    for (const auto&[str, freq] : word_counts) {
        if (str.empty()) {
            continue;
        }
        Node* head = nullptr;
        Node* tail = nullptr;

        for (const auto&c : str) {
            const int tokenId = static_cast<unsigned char>(c);
            const auto newNode = new Node();
            newNode->tokenId = tokenId;
            newNode->frequency = freq;
            newNode->next = nullptr;
            newNode->prev = tail;

            if (tail != nullptr) {
                tail->next = newNode;
            } else {
                head = newNode;
            }
            tail = newNode;
        }

        active_words.push_back(head);
    }
}

// size_t hash_pair(const int a, const int b) {
//     const size_t h1 = std::hash<int>()(a);
//     const size_t h2 = std::hash<int>()(b);
//
//     // Mix the bits using XOR and bit shifts
//     return h1 ^ (h2 << 1) ^ (h2 >> 1);
// }

inline uint64_t get_key(const int left_id, const int right_id) {
    // Shifts the left ID to the top 32 bits, and attaches the right ID to the bottom 32 bits.
    // Every single pair combination is now mathematically guaranteed to be unique.
    return (static_cast<uint64_t>(left_id) << 32) | static_cast<uint32_t>(right_id);
}

void Tokenizer::tally_pairs() {
    paired_frequencies.reserve(active_words.size() * 2);

    for (const auto&node : active_words) {
        if (node != nullptr) {
            auto curr = node;
            Node* prev = nullptr;
            while (curr != nullptr) {
                if (prev != nullptr) {
                    uint64_t hash = get_key(prev->tokenId, curr->tokenId);
                    paired_frequencies[hash].left_id = prev->tokenId;
                    paired_frequencies[hash].right_id = curr->tokenId;
                    paired_frequencies[hash].frequency += curr->frequency;
                }
                prev = curr;
                curr = curr->next;
            }
        }
    }

    std::vector<TokenPair> pairs_vec;
    pairs_vec.reserve(paired_frequencies.size());
    for (auto &token_pair: paired_frequencies | std::views::values) {
        pairs_vec.push_back(token_pair);
    }

    maxHeap = std::priority_queue(
        std::make_move_iterator(pairs_vec.begin()),
        std::make_move_iterator(pairs_vec.end())
    );
}

void Tokenizer::apply_merge(const int id_left, const int id_right, const int new_id) {
    const std::string new_word = reverse_vocab[id_left] + reverse_vocab[id_right];
    vocab[new_word] = new_id;
    reverse_vocab.push_back(new_word);

    for (const auto& head : active_words) {
        if (head == nullptr) continue;
        Node* curr = head;
        while (curr != nullptr && curr->next != nullptr) {
            if (curr->tokenId == id_left && curr->next->tokenId == id_right) {
                // merge here
                const auto right = curr->next;
                int word_freq = curr->frequency;

                uint64_t center_hash = get_key(id_left, id_right);
                paired_frequencies[center_hash].frequency -= word_freq;

                if (curr->prev != nullptr) {
                    uint64_t left_hash = get_key(curr->prev->tokenId, curr->tokenId);
                    paired_frequencies[left_hash].frequency -= word_freq;
                }

                if (right->next != nullptr) {
                    uint64_t right_hash = get_key(right->tokenId, right->next->tokenId);
                    paired_frequencies[right_hash].frequency -= word_freq;
                }

                curr->tokenId = new_id;
                curr->next = right->next;
                if (curr->next != nullptr) {
                    curr->next->prev = curr;
                }
                delete right; // We no longer need this node in this list

                if (curr->prev != nullptr) {
                    uint64_t left_hash = get_key(curr->prev->tokenId, curr->tokenId);
                    paired_frequencies[left_hash].left_id = curr->prev->tokenId;
                    paired_frequencies[left_hash].right_id = curr->tokenId;
                    paired_frequencies[left_hash].frequency += word_freq;
                }

                if (curr->next != nullptr) {
                    uint64_t right_hash = get_key(curr->tokenId, curr->next->tokenId);
                    paired_frequencies[right_hash].left_id = curr->tokenId;
                    paired_frequencies[right_hash].right_id = curr->next->tokenId;
                    paired_frequencies[right_hash].frequency += word_freq;
                }
            }
            curr = curr->next; // update the loop
        }
    }

    std::vector<TokenPair> pairs_vec;
    pairs_vec.reserve(paired_frequencies.size());
    for (auto &token_pair: paired_frequencies | std::views::values) {
        if (token_pair.frequency > 0) {
            pairs_vec.push_back(token_pair);
        }
    }
    maxHeap = std::priority_queue(
        std::make_move_iterator(pairs_vec.begin()),
        std::make_move_iterator(pairs_vec.end())
    );
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> final_tokens;

    // 1. Split the text into words/punctuation exactly like process_chunk
    std::string current_word;
    std::vector<std::string> words;
    for (const char c : text) {
        if (const auto uc = static_cast<unsigned char>(c); std::isalpha(uc) || std::isdigit(uc)) {
            current_word += c;
        } else {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word = "";
            }
            words.push_back(std::string(1, c));
        }
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
    }

    // 2. BPE Inference Loop
    for (const auto& w : words) {
        std::vector<int> tokens;
        for (char c : w) tokens.push_back(static_cast<unsigned char>(c));

        while (tokens.size() > 1) {
            int best_id = 1e9; // Start with an impossibly high ID
            int best_idx = -1;

            // Scan for the pair that was merged earliest in training (lowest vocab ID)
            for (size_t i = 0; i < tokens.size() - 1; i++) {
                std::string pair_str = reverse_vocab[tokens[i]] + reverse_vocab[tokens[i+1]];
                auto it = vocab.find(pair_str);

                // If the pair exists in our vocab and is older than our current best
                if (it != vocab.end() && it->second < best_id) {
                    best_id = it->second;
                    best_idx = i;
                }
            }

            if (best_idx == -1) break; // No more valid merges found

            // Execute the merge: update the left token, delete the right token
            tokens[best_idx] = best_id;
            tokens.erase(tokens.begin() + best_idx + 1);
        }
        // Append the compressed word to our final output
        final_tokens.insert(final_tokens.end(), tokens.begin(), tokens.end());
    }
    return final_tokens;
}

std::string Tokenizer::decode(const std::vector<int>& ids, const bool shouldSeparate) const {
    std::string text;
    for (const int id : ids) {
        if (shouldSeparate) text += "<";
        text += reverse_vocab[id];
        if (shouldSeparate) text += ">";
    }
    return text;
}

int Tokenizer::get_vocab_size() const {
    return vocab_size;
}

void Tokenizer::print_vocab() const {
    std::cout << "\n=== Learned Tokens ===" << std::endl;
    for (int i = INITIAL_VOCAB_SIZE; i < vocab_size; i++) {
        std::cout << "ID " << i << ": '" << reverse_vocab[i] << "'" << std::endl;
    }
}
