import numpy as np

# ── Sigmoid Activation & Derivative ──
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ── Dataset: AND Gate ──
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# ── Initialize Weights & Bias ──
np.random.seed(42)
weights = np.random.randn(2, 1) * 0.1    # 2 inputs → 1 output
bias = 0.0
lr = 0.5
epochs = 2000

# ── Predictions BEFORE Training ──
print("=" * 50)
print("   SINGLE NEURON MODEL (Manual, NumPy Only)")
print("=" * 50)
print("\n🔹 Predictions BEFORE training:")
z_before = np.dot(X, weights) + bias
pred_before = sigmoid(z_before)
for i in range(len(X)):
    print(f"   Input: {X[i]}  →  Raw: {z_before[i][0]:7.4f}  →  Sigmoid: {pred_before[i][0]:.4f}  →  Class: {int(pred_before[i][0] >= 0.5)}")

# ── Training (Gradient Descent) ──
loss_history = []
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)
    
    # Compute loss (Binary Cross-Entropy)
    loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
    loss_history.append(loss)
    
    # Backward pass (manual gradients)
    dz = output - y                      # dL/dz
    dw = np.dot(X.T, dz) / len(X)        # dL/dw
    db = np.mean(dz)                     # dL/db
    
    # Update weights
    weights -= lr * dw
    bias -= lr * db

# ── Predictions AFTER Training ──
print(f"\n🔹 Training complete ({epochs} epochs)")
print(f"   Final Loss: {loss_history[-1]:.6f}")
print(f"   Learned Weights: [{weights[0][0]:.4f}, {weights[1][0]:.4f}]")
print(f"   Learned Bias:    {bias:.4f}")

print("\n🔹 Predictions AFTER training:")
z_after = np.dot(X, weights) + bias
pred_after = sigmoid(z_after)
for i in range(len(X)):
    print(f"   Input: {X[i]}  →  Raw: {z_after[i][0]:7.4f}  →  Sigmoid: {pred_after[i][0]:.4f}  →  Class: {int(pred_after[i][0] >= 0.5)}")
print("=" * 50)
