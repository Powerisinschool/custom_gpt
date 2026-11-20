import torch
import torch.nn as nn
import numpy as np

print("--- STARTING PYTORCH MICRO-TRAINING ---")

# 1. System Parameters (Matching your C++ Engine)
d_model = 768

# PyTorch's version of your C++ W_q matrix
# (bias=False because you didn't add biases to Q, K, V in your C++ code)
W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

# An optimizer to handle the calculus (Backpropagation)
optimizer = torch.optim.Adam(W_q.parameters(), lr=0.01)

# 2. The Dummy Data
# We create 5 random input tokens, and a target output we want the AI to learn
dummy_input = torch.randn(5, d_model)
target_output = torch.randn(5, d_model)

# 3. The Training Loop
for epoch in range(1, 11):
    optimizer.zero_grad()             # Clear old math

    predictions = W_q(dummy_input)    # The Forward Pass (Matrix Multiplication)

    # Calculate how wrong the predictions are (Mean Squared Error for simplicity)
    loss = torch.nn.functional.mse_loss(predictions, target_output)

    loss.backward()                   # The Backward Pass (Calculus Derivatives)
    optimizer.step()                  # Update the weights

    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

print("\n--- EXPORTING TO C++ ENGINE ---")

# 4. The Transpose Trap (CRITICAL)
# PyTorch stores linear weights internally as [out_features, in_features].
# Your C++ cblas_sgemm expects [in_features, out_features].
# We MUST transpose (.T) before exporting, or your C++ engine will multiply garbage.
raw_weights = W_q.weight.detach().T

# 5. The Binary Export
# Convert to 32-bit floats, ensure memory is contiguous, and dump to a file
flat_array = raw_weights.contiguous().numpy().astype(np.float32).flatten()

filepath = "w_q_trained.bin"
flat_array.tofile(filepath)

print(f"Success! {len(flat_array)} floats exported to {filepath}.")
