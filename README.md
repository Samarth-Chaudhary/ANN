# Neural Network Fundamentals Lab

## Lab 1: Perceptron (OR Gate) & Lab 2: Sigmoid Neuron with Gradient Descent (AND Gate)

This repository contains two Jupyter notebooks demonstrating two core building blocks
of neural networks, both implemented from scratch using only NumPy:

- **`ANN_lab1_OR_perceptron.ipynb`** — a classic Perceptron trained on the OR gate using
  the Perceptron Learning Rule.
- **`ANN_lab2_AND_sigmoid.ipynb`** — a single Sigmoid neuron trained on the AND gate
  using manual backpropagation / gradient descent on Binary Cross-Entropy loss.

Each notebook includes the code, its actual run output, and a written
**Observation/Interpretation** section analyzing the results.

---

## 📌 Table of Contents

1. [Lab 1: Perceptron (OR Gate)](#-lab-1-perceptron-or-gate)
2. [Lab 2: Sigmoid Neuron with Gradient Descent (AND Gate)](#-lab-2-sigmoid-neuron-with-gradient-descent-and-gate)
3. [Comparison: Lab 1 vs Lab 2](#-comparison-lab-1-vs-lab-2)
4. [Viva Q&A](#-viva-qa)

---

## 🧠 Lab 1: Perceptron (OR Gate)

### Theory

A **Perceptron** is the simplest artificial neuron — a linear binary classifier. It
computes a weighted sum of its inputs plus a bias, then applies a **step activation
function**:

- Weighted sum: `z = w·x + b`
- Step activation: `f(z) = 1 if z ≥ 0, else 0`
- Perceptron Learning Rule (applied per sample): 
  - `error = target − output`
  - `w ← w + η · error · x`
  - `b ← b + η · error`

The OR gate is **linearly separable** — a single straight line can separate the
`(0,0) → 0` case from the three `→ 1` cases — so a Perceptron is guaranteed to find a
solution.

### Code Summary

- `step(x)` — vectorized step activation using `np.where`.
- `train_perceptron(X, y, lr, epochs)` — initializes weights/bias to zero, then loops
  over epochs and samples, updating weights whenever a prediction is wrong, and
  recording the total number of errors per epoch.
- Trained for **10 epochs** at **lr = 0.1** on the OR truth table.

### Observation / Interpretation

- The error count sequence was **`[2, 2, 1, 0, 0, 0, 0, 0, 0, 0]`** — the model made
  mistakes in epoch 1 and 2, dropped to 1 error in epoch 3, and reached **zero errors
  from epoch 4 onward**. This shows fast convergence, as expected for a simple,
  linearly separable 2-input problem.
- The final learned parameters were **weights = `[0.1, 0.1]`**, **bias = `-0.1`**. This
  defines the decision boundary `x1 + x2 = 1`: any point on or above this line is
  classified `1`.
- Final predictions `[0, 1, 1, 1]` are an exact match with the OR truth table — **100%
  training accuracy**.
- The two weights ending up equal reflects the symmetry of the OR function itself
  (swapping `x1` and `x2` doesn't change the output).
- Since convergence happened well before epoch 10, this confirms the OR gate is easily
  learnable by a single-layer Perceptron — consistent with the theoretical guarantee
  that Perceptrons converge on any linearly separable dataset.

---

## 🔢 Lab 2: Sigmoid Neuron with Gradient Descent (AND Gate)

### Theory

This lab trains a single neuron using a smooth, differentiable **Sigmoid activation**
instead of the step function, allowing it to be optimized with **gradient descent**:

- Sigmoid: `σ(z) = 1 / (1 + e^(−z))`
- Loss — Binary Cross-Entropy: `L = −(1/N) Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]`
- Because Sigmoid + BCE combine so that `∂L/∂z = ŷ − y`, the manual backward pass
  simplifies to:
  - `dz = output − y`
  - `dw = Xᵀ·dz / N`
  - `db = mean(dz)`
  - `w ← w − η·dw`, `b ← b − η·db`

### Code Summary

- Weights initialized with small random values (`np.random.randn(2,1) * 0.1`, seeded for
  reproducibility), bias initialized to `0.0`.
- Trained for **2000 epochs** at **lr = 0.5** on the AND truth table.
- Predictions are shown both **before** and **after** training for comparison.

### Observation / Interpretation

- **Before training**, with near-zero random weights, every output sigmoid value sat
  close to **0.50** (`0.5000, 0.4965, 0.5124, 0.5090`) — the model is essentially
  guessing, and it misclassifies two of the four inputs.
- **During training**, the Binary Cross-Entropy loss dropped steadily over 2000 epochs
  to a **final loss of ≈ 0.0174**, showing the model's confidence is converging
  strongly toward the correct labels rather than just crossing the 0.5 threshold.
- **After training**, weights grew to **≈ `[7.41, 7.41]`** with bias **≈ `-11.29`**.
  Both weights are large, positive, and equal — meaning the neuron only "fires" when
  `x1 + x2` is large enough to overcome the strongly negative bias, i.e. only when
  **both inputs are 1** — the correct AND logic.
- **Final predictions** were `[0, 0, 0, 1]`, an exact match with the AND truth table.
- Post-training output probabilities were sharply polarized — **≈ `0.0000, 0.0203,
  0.0203, 0.9716`** — rather than sitting near the 0.5 boundary. This shows the model
  isn't just barely correct, it's **confidently correct**, which is exactly what
  minimizing cross-entropy loss over many epochs is expected to produce.
- This demonstrates gradient descent successfully solving a linearly separable problem
  (AND) using a smooth activation and loss-based optimization, in contrast to Lab 1's
  discrete perceptron rule.

---

## ⚖️ Comparison: Lab 1 vs Lab 2

| Aspect | Lab 1 (Perceptron) | Lab 2 (Sigmoid Neuron) |
|---|---|---|
| Gate learned | OR | AND |
| Activation | Step (hard 0/1) | Sigmoid (smooth, 0 to 1) |
| Output type | Discrete class | Probability |
| Update rule | Perceptron rule (`error × input`) | Gradient descent on BCE loss |
| Differentiable? | No | Yes |
| Epochs needed | 10 (converged by epoch 4) | 2000 |
| Update frequency | Per sample (online) | Per epoch, using all samples (batch) |
| Generalizes to deep nets? | No — dead end | Yes — foundation of backpropagation |

---

## 🎓 Viva Q&A

**Q1: Why does the Perceptron converge so quickly on OR (Lab 1)?**  
A: OR is linearly separable with a wide margin between classes, and the Perceptron
Convergence Theorem guarantees a solution is found in a finite number of steps for any
linearly separable dataset.

**Q2: Why do we need 2000 epochs in Lab 2 versus only 10 in Lab 1?**  
A: Gradient descent takes small, proportionally-sized steps that shrink as the
model approaches the correct answer, so it needs many iterations to fully converge —
unlike the perceptron rule, which makes a full corrective jump on every misclassified
sample.

**Q3: Why are the final weights so large in Lab 2 (~7.4) compared to Lab 1 (~0.1)?**  
A: Sigmoid saturates — outputs close to `0` or `1` only happen when `|z|` is large. To
drive confident predictions (like 0.97 or 0.02), gradient descent pushes the weights and
bias to large magnitudes so the pre-activation `z` is strongly positive or negative.

**Q4: What does the loss value of 0.0174 actually tell us?**  
A: It's a low cross-entropy loss, meaning the model's predicted probabilities are very
close to the true 0/1 labels across all 4 training samples — indicating the model has
essentially solved the AND gate with high confidence, not just crossed the 0.5 threshold.

**Q5: Could either of these single-neuron models learn XOR?**  
A: No. Both models compute a linear decision boundary (`w·x + b`) before applying their
activation function. XOR is not linearly separable, so neither a step-function
perceptron nor a single sigmoid neuron can solve it — a hidden layer (multi-layer
network) is required.

**Q6: In Lab 2, why is `dz = output - y` used instead of calling `sigmoid_derivative()`?**  
A: When Sigmoid activation is paired with Binary Cross-Entropy loss, the chain rule
simplifies algebraically so the sigmoid derivative term cancels out, leaving this clean
expression — this is why `sigmoid_derivative()` is defined but never actually called in
the training loop.

**Q7: What would happen if Lab 1's OR gate were replaced with XOR?**  
A: The perceptron would never converge — the error count would keep oscillating and
never reach zero, because no straight-line decision boundary can separate XOR's classes.
