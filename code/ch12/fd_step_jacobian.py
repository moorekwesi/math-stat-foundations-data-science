# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Jacobian of softmax and of a composite map; finite-difference step size.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def num_jac(f, x, h=1e-6):
    """Central-difference Jacobian (numerator layout: m x n)."""
    cols = [(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(x.size)]
    return np.column_stack(cols)


# 1. softmax: J = diag(s) - s s'
z = rng.normal(size=4)
s = softmax(z)
J_exact = np.diag(s) - np.outer(s, s)
print("softmax Jacobian error   :",
      f"{np.max(np.abs(num_jac(softmax, z) - J_exact)):.2e}")
print("eigenvalues of J (PSD)   :", np.round(np.linalg.eigvalsh(J_exact), 4) + 0.0)

# 2. chain rule for g(x) = C tanh(Dx):  J = C diag(1 - tanh^2(Dx)) D
C = rng.normal(size=(2, 3))
D = rng.normal(size=(3, 5))
x = rng.normal(size=5)
g = lambda v: C @ np.tanh(D @ v)
J_chain = C @ np.diag(1 - np.tanh(D @ x) ** 2) @ D
print("chain-rule Jacobian shape:", J_chain.shape,
      f" error {np.max(np.abs(num_jac(g, x) - J_chain)):.2e}")

# 3. error of forward and central differences as the step h shrinks
f = lambda v: np.log(np.sum(np.exp(v)))        # log-sum-exp, gradient = s
hs = np.logspace(-14, -1, 40)
err_fwd, err_ctr = [], []
for h in hs:
    gf = np.array([(f(z + h * e) - f(z)) / h for e in np.eye(4)])
    gc = np.array([(f(z + h * e) - f(z - h * e)) / (2 * h) for e in np.eye(4)])
    err_fwd.append(np.max(np.abs(gf - s)))
    err_ctr.append(np.max(np.abs(gc - s)))
i_f, i_c = np.argmin(err_fwd), np.argmin(err_ctr)
print(f"best h forward: {hs[i_f]:.1e} (error {err_fwd[i_f]:.1e})")
print(f"best h central: {hs[i_c]:.1e} (error {err_ctr[i_c]:.1e})")

plt.figure(figsize=(6, 3.8))
plt.loglog(hs, err_fwd, "o-", ms=3, label="forward difference")
plt.loglog(hs, err_ctr, "s-", ms=3, label="central difference")
plt.xlabel("step size h")
plt.ylabel("max abs gradient error")
plt.title("Truncation vs rounding error (log-sum-exp)")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.savefig("ch12_fd_error.pdf", bbox_inches="tight")
