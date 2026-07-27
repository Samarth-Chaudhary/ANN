# Simulating a Perceptron using NumPy

import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.weights = np.zeros(n_inputs)   # start with weights = 0
        self.bias = 0                       # start with bias = 0
        self.lr = lr                        # learning rate

    def activate(self, z):
        return 1 if z >= 0 else 0           # step function

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activate(z)

    def fit(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                # update rule: only changes weights when prediction is wrong
                self.weights += self.lr * error * xi
                self.bias += self.lr * error


# ---- Data: AND gate ----
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# ---- Train ----
model = Perceptron(n_inputs=2)
model.fit(X, y, epochs=10)

# ---- Test ----
print("Weights:", model.weights, "| Bias:", model.bias)
for xi in X:
    print(f"Input: {xi} -> Predicted: {model.predict(xi)}")
