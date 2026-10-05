# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Leverage h_ii = diagonal of the hat matrix, computed stably from QR
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n = 200
hh_size = rng.poisson(4, size=n) + 1.0             # simulated household size
income = 2.0 + 0.3 * hh_size + rng.normal(0, 0.5, n)
hh_size[0], income[0] = 25.0, 4.0                  # one atypical household
X = np.column_stack([np.ones(n), hh_size])
k = X.shape[1]

Q, R = np.linalg.qr(X)                             # P = Q Q^T
h = np.sum(Q ** 2, axis=1)                         # h_ii = ||q_i||^2
print(f"sum of leverages = {h.sum():.4f}  (= k = {k})")
print(f"min h = {h.min():.4f} (>= 1/n = {1 / n:.4f}),  max h = {h.max():.4f}")
print(f"average k/n = {k / n:.4f};  rule of thumb flags h > 2k/n = {2 * k / n:.3f}")
print(f"flagged observations: {np.flatnonzero(h > 2 * k / n)}")

b_all = np.linalg.lstsq(X, income, rcond=None)[0]
b_drop = np.linalg.lstsq(X[1:], income[1:], rcond=None)[0]
print(f"\nslope with obs 0    = {b_all[1]:.4f}")
print(f"slope without obs 0 = {b_drop[1]:.4f}")

# Leave-one-out residual from a single fit: e_i / (1 - h_ii)
e = income - X @ b_all
loo_formula = e[0] / (1 - h[0])
loo_direct = income[0] - X[0] @ b_drop
print(f"LOO residual: formula {loo_formula:.4f}, refit {loo_direct:.4f}")

# A dummy for a region with a single observation has leverage exactly one
Xd = np.column_stack([X, np.eye(n)[:, 5]])
hd = np.sum(np.linalg.qr(Xd)[0] ** 2, axis=1)
print(f"\nleverage of the lone obs in its own dummy: h = {hd[5]:.6f}")
