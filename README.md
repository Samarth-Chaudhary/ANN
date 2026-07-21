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

### How it works

A **Perceptron** looks at its inputs, multiplies each by a weight, adds a bias, and
checks whether the total is positive or negative. If positive (or zero), it fires `1`;
otherwise it outputs `0`. After every wrong guess, it nudges its weights and bias a
little in the direction that would have made the guess correct. This is repeated for
several passes (epochs) over the whole dataset.

The OR gate is easy for this model because you can draw one straight line that puts
`(0,0)` on one side and the other three points on the other side.

### Code Summary

- `step(x)` — vectorized step activation using `np.where`.
- `train_perceptron(X, y, lr, epochs)` — initializes weights/bias to zero, then loops
  over epochs and samples, updating weights whenever a prediction is wrong, and
  recording the total number of errors per epoch.
- Trained for **10 epochs** at **lr = 0.1** on the OR truth table.

### Observation / Interpretation (actual run)

- The error count sequence was **`[2, 2, 1, 0, 0, 0, 0, 0, 0, 0]`** — mistakes in epochs
  1–2, one mistake in epoch 3, then **zero errors from epoch 4 onward**.
- Final weights = **`[0.1, 0.1]`**, bias = **`-0.1`**, and predictions `[0, 1, 1, 1]`
  exactly match the OR truth table — **100% training accuracy**.

### 🔬 What Happens If... (sensitivity experiments)

**What if the learning rate changes?**
Tried `lr = 0.01`, `0.1`, and `1.0`, all for 10 epochs on OR:

| lr | Final weights | Final bias | Error history |
|---|---|---|---|
| 0.01 | `[0.01, 0.01]` | `-0.01` | `[2,2,1,0,0,0,0,0,0,0]` |
| 0.1  | `[0.1, 0.1]`   | `-0.1`  | `[2,2,1,0,0,0,0,0,0,0]` |
| 1.0  | `[1.0, 1.0]`   | `-1.0`  | `[2,2,1,0,0,0,0,0,0,0]` |

The error history is **identical** in all three cases — only the final weight/bias
*magnitude* scales up or down with `lr`. This happens because the perceptron update is
always `lr × error × input`, and `error` is always exactly `-1`, `0`, or `+1` here — so
changing `lr` just rescales the step size uniformly without changing the direction or
number of corrective steps needed. For this particular simple, well-separated dataset,
the learning rate barely matters. (On messier/real-valued data, too high a learning rate
can cause oscillation instead.)

**What if we train for fewer epochs?**

| Epochs | Final weights | Final bias | Errors this run |
|---|---|---|---|
| 1  | `[0.0, 0.1]`  | `0.0`   | `[2]` (still wrong) |
| 3  | `[0.1, 0.1]`  | `-0.1`  | `[2,2,1]` (just converged) |
| 10 | `[0.1, 0.1]`  | `-0.1`  | `[2,2,1,0,0,0,0,0,0,0]` (converged + confirmed stable) |

Stopping at epoch 1 leaves the model still wrong. By epoch 3 it has already found the
correct boundary — the extra epochs (4–10) don't change the weights at all, they simply
confirm the model has stabilized. This shows why it's useful to track the error history:
it tells you exactly when you can safely stop training.

**What if we swap OR for XOR (a non-linearly-separable gate)?**

| Epochs | Final weights | Final bias | Error history |
|---|---|---|---|
| 10 | `[-0.1, 0.0]` | `0.0` | `[3,3,4,4,4,4,4,4,4,4]` |
| 50 | `[-0.1, 0.0]` | `0.0` | errors stay stuck at `4` for every remaining epoch |

Unlike OR, the error count **never reaches zero** — no matter how many epochs we run, it
plateaus at a constant number of mistakes. This is the practical signature of trying to
fit a linearly-separable-only model (the perceptron) to a problem that isn't linearly
separable. If you ever see the error history flatten out at a non-zero value instead of
hitting zero, that's a strong signal the model's decision boundary is fundamentally the
wrong shape for the data — more epochs will not fix it.

---

## 🔢 Lab 2: Sigmoid Neuron with Gradient Descent (AND Gate)

### How it works

This neuron gives a confidence score between 0 and 1 (via the Sigmoid function) instead
of a hard 0/1 answer. It's graded using a loss function that punishes confident wrong
answers heavily. After each pass over the data, it computes how far off its predictions
were and nudges the weights and bias a small step in the direction that reduces that
loss. This is repeated many times (epochs), and because the steps are small and smooth,
it typically needs far more epochs than the perceptron to fully settle.

### Code Summary

- Weights initialized with small random values (`np.random.randn(2,1) * 0.1`, seeded for
  reproducibility), bias initialized to `0.0`.
- Trained for **2000 epochs** at **lr = 0.5** on the AND truth table.
- Predictions are shown both **before** and **after** training for comparison.

### Observation / Interpretation (actual run)

- **Before training**, every output sat close to **0.50** (`0.5000, 0.4965, 0.5124,
  0.5090`) — the model was essentially guessing, misclassifying two of the four inputs.
- **After 2000 epochs**, the loss dropped to **≈ 0.0174**, weights grew to **≈ `[7.41,
  7.41]`**, bias to **≈ `-11.29`**, and predictions became `[0, 0, 0, 1]` — an exact
  match with AND, with confident, polarized probabilities (`0.0000, 0.0203, 0.0203,
  0.9716`) rather than borderline ones.

### 🔬 What Happens If... (sensitivity experiments)

**What if the learning rate changes?** (2000 epochs each)

| lr | Final loss | Final weights | Final bias |
|---|---|---|---|
| 0.01 | 0.3627 | `[1.04, 1.02]` | `-1.93` |
| 0.5  | 0.0174 | `[7.41, 7.41]` | `-11.29` |
| 5.0  | 0.0017 | `[12.08, 12.08]` | `-18.28` |

A **too-small** learning rate (`0.01`) leaves the loss far higher after the same number
of epochs — the model hasn't had time to grow confident weights yet. A **larger**
learning rate (`5.0`) reaches an even lower loss in the same 2000 epochs, with much
larger (more confident/extreme) weights. In this simple problem, a bigger learning rate
sped things up — but pushed too far, it can misbehave (see next experiment).

**What if the learning rate is pushed too high (instability check)?**  
Looking at the very first few loss values (epoch-by-epoch) at 50 epochs:

| lr | Loss at epoch 1,2,3,4,5 |
|---|---|
| 5.0  | 0.693 → 0.565 → 0.464 → 0.402 → 0.355 (steadily decreasing) |
| 20.0 | 0.693 → **1.303 → 2.497 → 3.119** → 0.062 (loss spikes up before crashing back down) |

At `lr = 20`, the loss actually **increases** for a few epochs (from 0.69 up to over 3.0)
before eventually recovering — a clear sign of instability/overshooting: the weight
update step became so large it jumped past the good solution and had to correct course.
This is the practical demonstration of "too high a learning rate causes oscillation."

**What if we train for fewer epochs?** (lr = 0.5)

| Epochs | Final loss | Final weights | Final bias |
|---|---|---|---|
| 10   | 0.5461 | `[0.24, 0.19]` | `-0.82` |
| 100  | 0.2287 | `[2.04, 2.04]` | `-3.32` |
| 2000 | 0.0174 | `[7.41, 7.41]` | `-11.29` |

The loss keeps dropping the longer we train, but with **diminishing returns** — going
from 10 → 100 epochs cuts the loss by more than half, while going from 100 → 2000
epochs (20× more training) also helps a lot, but the weights are already "confident
enough" that predictions would already threshold correctly well before epoch 2000. This
shows that near-perfect probability separation (like 0.97 vs 0.02) needs many more
epochs than merely crossing the 0.5 decision threshold.

**What if the initial random weights are much larger?**  
Compared `init_scale = 0.1` (used in the code) vs `init_scale = 5.0`:

| Init scale | Final loss | Final weights | Final bias |
|---|---|---|---|
| 0.1 | 0.017437 | `[7.4127, 7.4127]` | `-11.2916` |
| 5.0 | 0.017311 | `[7.4274, 7.4274]` | `-11.3136` |

The final result is almost identical either way. For a single neuron on a tiny,
simple, linearly separable dataset like AND, the starting point barely matters —
gradient descent finds essentially the same solution regardless. (This would **not** be
true in deep, multi-layer networks, where large or all-equal initial weights can cause
serious training problems.)

---

## ⚖️ Comparison: Lab 1 vs Lab 2

| Aspect | Lab 1 (Perceptron) | Lab 2 (Sigmoid Neuron) |
|---|---|---|
| Gate learned | OR | AND |
| Activation | Step (hard 0/1) | Sigmoid (smooth, 0 to 1) |
| Output type | Discrete class | Probability |
| Update rule | Perceptron rule (`error × input`) | Gradient descent on cross-entropy loss |
| Sensitive to learning rate? | Barely, for this simple data | Yes — too high causes loss spikes/instability |
| Sensitive to epochs? | No — converges by epoch 3-4 | Yes — loss keeps improving up to 2000 epochs |
| Sensitive to init weights? | N/A (always starts at zero) | No, for this simple problem |
| Can it solve XOR? | No — gets permanently stuck (errors plateau) | No — still a linear boundary underneath |
| Generalizes to deep nets? | No — dead end | Yes — foundation of backpropagation |

---

## 🎓 Viva Q&A

**Q1: Why does the Perceptron converge so quickly on OR (Lab 1)?**  
A: OR is linearly separable with a wide margin between classes, so the model finds a
correct boundary in just a few passes over the data.

**Q2: Why do we need 2000 epochs in Lab 2 versus only 10 in Lab 1?**  
A: Gradient descent takes small steps that shrink as the model gets closer to the right
answer, so it needs many iterations to fully settle — unlike the perceptron rule, which
makes a full corrective jump on every misclassified sample.

**Q3: Why are the final weights so large in Lab 2 (~7.4) compared to Lab 1 (~0.1)?**  
A: The sigmoid function only gets close to 0 or 1 when its input is a large positive or
negative number. To make confident predictions (like 0.97 or 0.02), gradient descent has
to push the weights and bias to larger magnitudes.

**Q4: What does a loss value of 0.0174 actually tell us?**  
A: It's a very low cross-entropy loss, meaning the model's predicted probabilities are
very close to the true 0/1 labels — the model isn't just crossing the 0.5 threshold, it's
confidently correct.

**Q5: Could either of these single-neuron models learn XOR?**  
A: No — both models draw a single straight decision line before applying their
activation function. XOR needs a bent/non-linear boundary, so a hidden layer (multi-layer
network) is required. Our own experiment above confirms this: the perceptron's error
count never reaches zero on XOR, no matter how many epochs it's given.

**Q6: What happens if the learning rate in Lab 2 is set too high?**  
A: As shown above, at `lr = 20` the loss actually spikes upward for the first few epochs
before crashing back down — the update step overshoots the good solution repeatedly
before eventually correcting itself. If pushed even further, this can cause the loss to
diverge entirely instead of recovering.

**Q7: Does the starting learning rate matter for the Perceptron in Lab 1?**  
A: Not for the error history or convergence speed here — only the *magnitude* of the
final weights/bias scales with the learning rate, since every error is always exactly
`-1`, `0`, or `+1` on this dataset.

**Q8: Does weight initialization matter for Lab 2?**  
A: For this single neuron on a tiny dataset, barely — both a small and a much larger
random initialization converge to nearly the same final weights and loss.
