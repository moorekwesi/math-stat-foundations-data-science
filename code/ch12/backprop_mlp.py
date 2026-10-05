# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Two-layer neural network with hand-written backpropagation on sklearn digits.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(803)
X, y = load_digits(return_X_y=True)
X = X / 16.0                                  # pixel intensities in [0, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=803)
Ytr = np.eye(10)[ytr]
d, H, K = X.shape[1], 32, 10


def init():
    return {"W1": rng.normal(0, 1 / np.sqrt(d), (d, H)), "b1": np.zeros(H),
            "W2": rng.normal(0, 1 / np.sqrt(H), (H, K)), "b2": np.zeros(K)}


def forward_backward(P, X, Y, lam=1e-4):
    n = X.shape[0]
    A1 = np.tanh(X @ P["W1"] + P["b1"])                 # hidden layer
    Z2 = A1 @ P["W2"] + P["b2"]
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)
    Q = np.exp(Z2)
    Q /= Q.sum(axis=1, keepdims=True)                   # softmax output
    loss = -np.sum(Y * np.log(Q)) / n
    loss += 0.5 * lam * (np.sum(P["W1"]**2) + np.sum(P["W2"]**2))
    # ---- backward pass: adjoints flow from the loss to every parameter
    dZ2 = (Q - Y) / n                                   # d loss / d Z2
    g = {"W2": A1.T @ dZ2 + lam * P["W2"], "b2": dZ2.sum(axis=0)}
    dA1 = dZ2 @ P["W2"].T
    dZ1 = dA1 * (1 - A1**2)                             # tanh' = 1 - tanh^2
    g["W1"] = X.T @ dZ1 + lam * P["W1"]
    g["b1"] = dZ1.sum(axis=0)
    return loss, g, Q


# ---- gradient check on a small batch
P = init()
Xb, Yb = Xtr[:20], Ytr[:20]
_, g, _ = forward_backward(P, Xb, Yb)
num_v, ana_v = [], []
for name in P:
    for idx in [tuple(rng.integers(0, s) for s in P[name].shape)
                for _ in range(5)]:
        old = P[name][idx]
        P[name][idx] = old + 1e-6
        lp = forward_backward(P, Xb, Yb)[0]
        P[name][idx] = old - 1e-6
        lm = forward_backward(P, Xb, Yb)[0]
        P[name][idx] = old
        num_v.append((lp - lm) / 2e-6)
        ana_v.append(g[name][idx])
num_v, ana_v = np.array(num_v), np.array(ana_v)
worst = np.linalg.norm(num_v - ana_v) / np.linalg.norm(num_v + ana_v)
print(f"gradient check (20 entries): relative error = {worst:.1e}")

# ---- mini-batch SGD with momentum
P = init()
V = {k: np.zeros_like(v) for k, v in P.items()}
lr, mom, epochs, bs = 0.5, 0.9, 60, 64
hist_tr, hist_te = [], []
for ep in range(epochs):
    perm = rng.permutation(len(Xtr))
    for i in range(0, len(Xtr), bs):
        idx = perm[i:i + bs]
        _, g, _ = forward_backward(P, Xtr[idx], Ytr[idx])
        for k in P:
            V[k] = mom * V[k] - lr * g[k]
            P[k] += V[k]
    hist_tr.append(forward_backward(P, Xtr, Ytr)[0])
    hist_te.append(forward_backward(P, Xte, np.eye(10)[yte])[0])
    if ep % 10 == 0 or ep == epochs - 1:
        print(f"epoch {ep:>3}: train loss {hist_tr[-1]:.4f}  "
              f"test loss {hist_te[-1]:.4f}")

acc = lambda Xs, ys: np.mean(forward_backward(P, Xs, np.eye(10)[ys])[2]
                             .argmax(1) == ys)
print(f"accuracy: train {acc(Xtr, ytr):.3f}, test {acc(Xte, yte):.3f}")
n_par = sum(v.size for v in P.values())
print(f"parameters: {n_par}; one backward pass gives all {n_par} partials")

plt.figure(figsize=(6, 3.8))
plt.semilogy(hist_tr, label="training loss")
plt.semilogy(hist_te, label="test loss")
plt.xlabel("epoch")
plt.ylabel("cross-entropy (log scale)")
plt.title("Two-layer network on digits, hand-coded backprop")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.savefig("ch12_mlp_loss.pdf", bbox_inches="tight")
