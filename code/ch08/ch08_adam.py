# DEP 803 -- Chapter 8: Optimisation
# Momentum, AdaGrad, RMSProp and Adam in NumPy; comparison with sklearn
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression

rng = np.random.default_rng(803)
n, p = 50_000, 8
X = rng.normal(size=(n, p)) * np.array([1, 1, 1, 5, 5, 20, 0.2, 0.2])
X = np.column_stack([np.ones(n), X])                 # badly scaled features
beta = np.array([-0.5, 0.8, -0.6, 0.4, 0.1, -0.1, 0.02, 3.0, -2.0])
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta))).astype(float)


def loss(b):
    z = X @ b
    return np.mean(np.logaddexp(0, z) - y * z)


def grad(b, idx):
    return X[idx].T @ (1 / (1 + np.exp(-X[idx] @ b)) - y[idx]) / idx.size


class Adam:
    """Kingma and Ba (2015) with bias correction."""
    def __init__(self, dim, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return -self.lr * mhat / (np.sqrt(vhat) + self.eps)


class Simple:
    """SGD, momentum, AdaGrad and RMSProp as one small class."""
    def __init__(self, dim, kind, lr):
        self.kind, self.lr = kind, lr
        self.v, self.s = np.zeros(dim), np.zeros(dim)

    def step(self, g):
        if self.kind == "SGD":
            return -self.lr * g
        if self.kind == "momentum":
            self.v = 0.9 * self.v - self.lr * g
            return self.v
        if self.kind == "AdaGrad":
            self.s += g**2
        else:                                        # RMSProp
            self.s = 0.9 * self.s + 0.1 * g**2
        return -self.lr * g / (np.sqrt(self.s) + 1e-8)


opts = {"SGD": lambda: Simple(p + 1, "SGD", 0.002),
        "momentum": lambda: Simple(p + 1, "momentum", 0.0002),
        "AdaGrad": lambda: Simple(p + 1, "AdaGrad", 0.1),
        "RMSProp": lambda: Simple(p + 1, "RMSProp", 0.005),
        "Adam": lambda: Adam(p + 1, lr=0.005)}
ref = LogisticRegression(penalty="none", fit_intercept=False, max_iter=5000,
                         tol=1e-10).fit(X, y)
f_star = loss(ref.coef_[0])
print(f"{'optimiser':<10}{'excess loss':>13}{'max |b - b*|':>14}")
plt.figure(figsize=(6, 3.8))
for name, make in opts.items():
    r = np.random.default_rng(7)
    opt, b, curve = make(), np.zeros(p + 1), []
    for ep in range(4):
        perm = r.permutation(n)
        for s in range(0, n, 100):
            b += opt.step(grad(b, perm[s:s + 100]))
            if (s // 100) % 25 == 0:
                curve.append(loss(b) - f_star)
    print(f"{name:<10}{loss(b) - f_star:>13.2e}"
          f"{np.abs(b - ref.coef_[0]).max():>14.4f}")
    plt.semilogy(np.abs(curve) + 1e-7, label=name, lw=1)
plt.gca().set(xlabel="minibatch update (x25), batch size 100",
              ylabel="excess training loss", title="Adaptive methods on a badly scaled logit")
plt.legend(frameon=False, ncol=2)
plt.savefig("ch08_adam.pdf", bbox_inches="tight")

sk = SGDClassifier(loss="log_loss", penalty=None, learning_rate="optimal",
                   fit_intercept=False, max_iter=20, tol=None,
                   random_state=0).fit(X, y)
print(f"sklearn SGDClassifier, raw X   : excess loss "
      f"{loss(sk.coef_[0]) - f_star:.2e}")
m, s = X[:, 1:].mean(0), X[:, 1:].std(0)            # standardise, refit
sk2 = SGDClassifier(loss="log_loss", penalty=None, max_iter=20, tol=None,
                    random_state=0).fit((X[:, 1:] - m) / s, y)
c = sk2.coef_[0] / s                                 # back to raw units
b_sk = np.concatenate([[sk2.intercept_[0] - c @ m], c])
print(f"sklearn SGDClassifier, scaled X: excess loss "
      f"{loss(b_sk) - f_star:.2e}")
