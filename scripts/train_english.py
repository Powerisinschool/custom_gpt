from collections import Counter
import json
import sys
sys.path.append("../build")
import custom_tokenizer
import torch
import torch.nn as nn
import numpy as np

print("--- INITIALIZING TINY GPT ---")

tokenizer = custom_tokenizer.Tokenizer()
tokenizer.train("../data/shakespeare.txt", 15000) # Train on Shakespeare's works, limit vocab to 5000 words

VOCAB_SIZE = tokenizer.vocab_size()
print(f"Vocabulary Size: {VOCAB_SIZE} words")

# tokenizer.print_vocab()

# 1. The Toy Dataset
# text = "the quick brown fox jumps over the lazy dog. the smart fox sleeps."
with open('../data/shakespeare.txt', 'r', encoding='utf-8') as file:
    data = file.read()
# words = text.split()
token_ids = tokenizer.encode(data)
json.dump(token_ids, open("./vocab.json", "w"))
# print(f"Tokens: {token_ids}")
# print(f"Decoded: {tokenizer.decode(token_ids)}")

print(f"First token: {token_ids[0]} | Decoded: '{tokenizer.decode([token_ids[0]])}'")
print(f"First 10 tokens: {token_ids[:10]}\nDecoded: '{tokenizer.decode(token_ids[:10])}'")
counts = Counter(token_ids)
print(f"Most common tokens: {counts.most_common(10)}")

# 2. System Parameters (Scaled down for instant CPU training)
d_model = 128 # 768
d_ff = 512 # 3072
seq_len = 1024

# 3. The PyTorch Architecture (Mirroring your C++)
embeddings = nn.Embedding(VOCAB_SIZE, d_model)
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)
W_o = nn.Linear(d_model, d_model, bias=False)
W_1 = nn.Linear(d_model, d_ff)
W_2 = nn.Linear(d_ff, d_model)
lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False) # Maps 64-dim back to words

optimizer = torch.optim.Adam(list(embeddings.parameters()) +
                             list(W_q.parameters()) + list(W_k.parameters()) +
                             list(W_v.parameters()) + list(W_o.parameters()) +
                             list(W_1.parameters()) + list(W_2.parameters()) +
                             list(lm_head.parameters()), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

# 4. The Training Loop
print("--- TRAINING COMMENCED ---")
for epoch in range(5000):
    # optimizer.zero_grad()

    # Create a random sequence of seq_len tokens from our text
    start_idx = np.random.randint(0, len(token_ids) - seq_len - 1)
    input_ids = token_ids[start_idx : start_idx + seq_len]
    target_ids = token_ids[start_idx + 1 : start_idx + seq_len + 1]

    x = torch.tensor([input_ids])     # Input
    y_true = torch.tensor([target_ids]) # Target

    # Forward Pass (Simplified for the script, PyTorch does the heavy lifting)
    emb = embeddings(x)

    norm1 = nn.LayerNorm(d_model)
    norm2 = nn.LayerNorm(d_model)

    num_layers = 4
    x = emb

    # q, k, v = W_q(emb), W_k(emb), W_v(emb)
    # # PyTorch's optimized attention function with layer normalization
    # attn_out = norm1(emb + torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))

    # # FFN & LM Head
    # ff_out = norm2(attn_out + W_2(torch.relu(W_1(attn_out))))

    for _ in range(num_layers):
        q, k, v = W_q(emb), W_k(emb), W_v(emb)
        attn_out = norm1(x + nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))
        x = norm2(attn_out + W_2(torch.relu(W_1(attn_out))))

    logits = lm_head(x) # Shape: [1, seq_len, vocab_size]

    # Calculate Cross-Entropy Loss
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y_true.view(-1))

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

print("--- EXPORTING KNOWLEDGE TO C++ ---")

# 5. The Binary Export (Remembering the Transpose Trap!)
def export_tensor(tensor, filename, transpose=True):
    if transpose:
        tensor = tensor.T
    flat = tensor.detach().contiguous().numpy().astype(np.float32).flatten()
    flat.tofile(f"../bin/{filename}")

export_tensor(embeddings.weight, "embeddings.bin", transpose=False)
export_tensor(W_q.weight, "w_q.bin")
export_tensor(W_k.weight, "w_k.bin")
export_tensor(W_v.weight, "w_v.bin")
export_tensor(W_o.weight, "w_o.bin")
export_tensor(W_1.weight, "w_1.bin")
export_tensor(W_2.weight, "w_2.bin")
export_tensor(lm_head.weight, "lm_head.bin")

# # Export the dictionary so C++ knows which ID matches which word
# with open("vocab.txt", "w") as f:
#     for word in vocab:
#         f.write(f"{word}\n")

print("Export complete. C++ engine is ready for inference.")
