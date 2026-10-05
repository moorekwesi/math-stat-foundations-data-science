# DEP 803 -- Chapter 8: Optimisation
# Finite-difference gradients and gradient checking for the logit loss
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n, p = 500, 4
X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
beta_true = np.array([-0.5, 1.0, -0.8, 0.3])
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta_true))).astype(float)


def loss(b):
    """Average negative log-likelihood of the logit model."""
    z = X @ b
    return np.mean(np.logaddexp(0, z) - y * z)


def grad(b):
    """Analytic gradient: X'(sigma(Xb) - y) / n."""
    return X.T @ (1 / (1 + np.exp(-X @ b)) - y) / n


def fd_grad(f, b, h, central=True):
    g = np.zeros_like(b)
    for j in range(b.size):
        e = np.zeros_like(b)
        e[j] = h
        g[j] = (f(b + e) - f(b - e)) / (2 * h) if central else \
            (f(b + e) - f(b)) / h
    return g


b = rng.normal(size=p)
ga = grad(b)
print("analytic gradient :", np.round(ga, 8))
print("central FD, h=1e-5:", np.round(fd_grad(loss, b, 1e-5), 8))
rel = np.linalg.norm(fd_grad(loss, b, 1e-5) - ga) / np.linalg.norm(ga)
print(f"relative error    : {rel:.2e}  (pass if < 1e-6)")


def bad_grad(b):          # a typical bug: forgot to divide by n
    return X.T @ (1 / (1 + np.exp(-X @ b)) - y)


rel_bad = np.linalg.norm(fd_grad(loss, b, 1e-5) - bad_grad(b)) / \
    np.linalg.norm(bad_grad(b))
print(f"buggy gradient    : relative error {rel_bad:.2e}  -> FAIL")

hs = np.logspace(-14, -1, 60)
err_f = [np.linalg.norm(fd_grad(loss, b, h, False) - ga) for h in hs]
err_c = [np.linalg.norm(fd_grad(loss, b, h, True) - ga) for h in hs]
print(f"best h forward : {hs[np.argmin(err_f)]:.1e}, error {min(err_f):.1e}")
print(f"best h central : {hs[np.argmin(err_c)]:.1e}, error {min(err_c):.1e}")

plt.figure(figsize=(6, 3.8))
plt.loglog(hs, err_f, "o-", ms=3, label="forward difference, O(h)")
plt.loglog(hs, err_c, "s-", ms=3, label="central difference, O(h^2)")
plt.gca().set(xlabel="step h", ylabel="error in gradient (2-norm)",
              title="Truncation error vs rounding error")
plt.legend(frameon=False)
plt.savefig("ch08_fd_error.pdf", bbox_inches="tight")
