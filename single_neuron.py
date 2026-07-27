# Single Neuron Model implemented manually using NumPy

import numpy as np

# ---- Activation function ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)   # derivative in terms of the sigmoid output itself

# ---- Dataset: AND gate ----
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# ---- Initialize a single neuron's parameters ----
np.random.seed(1)
weights = np.random.randn(2, 1)   # 2 inputs -> 1 neuron
bias = np.zeros((1,))
lr = 0.1

# ---- Predictions BEFORE training ----
print("Predictions BEFORE training:")
z = np.dot(X, weights) + bias
a = sigmoid(z)
for i in range(len(X)):
    print(f"Input: {X[i]} => Predicted: {round(float(a[i]), 4)} => Class: {int(a[i] >= 0.5)}")

# ---- Training loop (manual forward + backward pass) ----
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    a = sigmoid(z)

    # Loss (mean squared error, kept simple/readable)
    loss = np.mean((y - a) ** 2)

    # Backward pass (manual gradient computation)
    error = a - y
    d_a = error * sigmoid_derivative(a)
    d_weights = np.dot(X.T, d_a) / len(X)
    d_bias = np.mean(d_a)

    # Update parameters (gradient descent)
    weights -= lr * d_weights
    bias -= lr * d_bias

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- Predictions AFTER training ----
print("\nPredictions AFTER training:")
z = np.dot(X, weights) + bias
a = sigmoid(z)
for i in range(len(X)):
    print(f"Input: {X[i]} => Predicted: {round(float(a[i]), 4)} => Class: {int(a[i] >= 0.5)}")
