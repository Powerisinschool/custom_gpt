//
// Created by Tolulope Olagunju on 24/05/2026.
//

#ifndef CUSTOM_GPT_BPE_NODE_HPP
#define CUSTOM_GPT_BPE_NODE_HPP

struct Node {
    Node* prev;
    Node* next;
    int tokenId;
    int frequency;
};

#endif //CUSTOM_GPT_BPE_NODE_HPP
