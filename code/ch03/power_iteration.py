# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# Power iteration, Rayleigh quotient and inverse iteration; convergence rates.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)


def make_sym(eigs, rng):
    """Symmetric matrix with prescribed eigenvalues and random eigenvectors."""
    Q, _ = np.linalg.qr(rng.standard_normal((len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T, Q


def power_iteration(A, x0, iters):
    x = x0 / np.linalg.norm(x0)
    rq = []
    for _ in range(iters):
        y = A @ x
        x = y / np.linalg.norm(y)
        rq.append(x @ A @ x)              # Rayleigh quotient
    return x, np.array(rq)


n, iters = 50, 60
x0 = rng.standard_normal(n)
fig, ax = plt.subplots(figsize=(6, 3.8))
for ratio in (0.5, 0.8, 0.95):
    eigs = np.concatenate([[10.0, 10.0 * ratio],
                           rng.uniform(0, 10.0 * ratio * 0.9, n - 2)])
    A, Q = make_sym(eigs, rng)
    x, rq = power_iteration(A, x0, iters)
    err = np.abs(rq - 10.0)
    vec_err = 1 - abs(x @ Q[:, 0])
    ax.semilogy(err, lw=2, label=f"|l2/l1| = {ratio}")
    ax.semilogy(10 * ratio ** (2 * np.arange(iters)), "k:", lw=0.8)
    print(f"ratio {ratio:4.2f}: RQ error after {iters} its = {err[-1]:.2e}, "
          f"1-|cos| = {vec_err:.2e}")
ax.set_xlabel("iteration k")
ax.set_ylabel("|Rayleigh quotient - l1|")
ax.set_title("Power iteration: error ~ (l2/l1)^(2k) for symmetric A")
ax.legend(frameon=False)
plt.savefig("ch03_power_convergence.pdf", bbox_inches="tight")

# Inverse iteration with a shift: find the eigenvalue closest to mu
eigs = np.arange(1.0, n + 1.0)
A, Q = make_sym(eigs, rng)
mu = 17.3
x = x0 / np.linalg.norm(x0)
M = A - mu * np.eye(n)
for k in range(8):
    x = np.linalg.solve(M, x)
    x /= np.linalg.norm(x)
    print(f"inverse iteration k={k + 1}: RQ = {x @ A @ x:.12f}")

# Rayleigh quotient iteration: update the shift every step (cubic rate)
x = Q[:, 30] + 0.03 * rng.standard_normal(n)
x /= np.linalg.norm(x)
for k in range(5):
    rho = x @ A @ x
    print(f"RQI k={k}: rho = {rho:.15f}")
    x = np.linalg.solve(A - rho * np.eye(n), x)
    x /= np.linalg.norm(x)
