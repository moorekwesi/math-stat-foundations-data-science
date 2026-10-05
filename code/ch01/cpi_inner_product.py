# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# A Laspeyres consumer price index as an inner product (simulated basket).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

items = ["food", "alcohol/tobacco", "clothing", "housing/utilities",
         "furnishings", "health", "transport", "communication",
         "recreation", "education"]
# simulated base-period quantities and prices (illustrative, not official)
q0 = np.array([120, 6, 14, 30, 8, 5, 40, 12, 7, 4], dtype=float)
p0 = np.array([5.0, 12.0, 30.0, 25.0, 40.0, 60.0, 8.0, 15.0, 20.0, 150.0])
p1 = p0 * np.array([1.24, 1.10, 1.08, 1.30, 1.12, 1.15, 1.35, 1.05, 1.09, 1.20])
q1 = q0 * np.array([0.95, 1.00, 0.97, 0.96, 0.99, 1.01, 0.90, 1.05, 0.98, 1.00])

w = p0 * q0 / (p0 @ q0)              # base-period expenditure weights
r = p1 / p0                          # price relatives
L = 100 * (w @ r)                    # Laspeyres as a weighted mean of relatives
L2 = 100 * (p1 @ q0) / (p0 @ q0)     # same index as a ratio of inner products
P = 100 * (p1 @ q1) / (p0 @ q1)      # Paasche
F = np.sqrt(L * P)                   # Fisher ideal
print(f"sum of weights = {w.sum():.6f}")
print(f"Laspeyres  100 <w, r>             = {L:.3f}")
print(f"Laspeyres  100 <p1,q0>/<p0,q0>    = {L2:.3f}")
print(f"Paasche    100 <p1,q1>/<p0,q1>    = {P:.3f}")
print(f"Fisher     sqrt(L * P)            = {F:.3f}")
contrib = 100 * w * (r - 1)          # percentage-point contributions
print(f"\n{'item':18s} {'weight':>7s} {'relative':>9s} {'contrib(pp)':>12s}")
for k in np.argsort(-contrib):
    print(f"{items[k]:18s} {w[k]:7.3f} {r[k]:9.3f} {contrib[k]:12.3f}")
print(f"{'total inflation':18s} {'':7s} {'':9s} {contrib.sum():12.3f}")
