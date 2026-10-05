# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Identification: a programme effect that is collinear with region effects
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, G, tau = 1200, 16, 0.30            # households, regions, true effect
region = rng.integers(0, G, size=n)
treated_regions = np.arange(5)        # programme rolled out region-wide
T = np.isin(region, treated_regions).astype(float)
educ = rng.normal(9, 3, size=n)
alpha = rng.normal(0, 0.4, size=G)    # unobserved regional effects
y = 1.0 + tau * T + 0.08 * educ + alpha[region] + rng.normal(0, 0.5, n)

D = np.eye(G)[region]                 # all 16 region dummies, no constant
X = np.column_stack([T, educ, D])     # T = sum of first five dummy columns
print(f"columns = {X.shape[1]}, rank = {np.linalg.matrix_rank(X)}")

# Two different exact least-squares solutions
b1 = np.linalg.lstsq(X, y, rcond=None)[0]          # minimum-norm solution
c = np.r_[1.0, 0.0, -1.0 * np.isin(np.arange(G), treated_regions)]  # Xc=0
b2 = b1 + 2.0 * c
print(f"||X c|| = {np.linalg.norm(X @ c):.1e}")
print(f"tau-hat from solution 1 = {b1[0]: .4f}")
print(f"tau-hat from solution 2 = {b2[0]: .4f}")
print(f"max |fitted1 - fitted2| = {np.abs(X @ b1 - X @ b2).max():.1e}")

# An estimable function: l' beta is identified iff l is in row(X)
_, s, Vt = np.linalg.svd(X, full_matrices=False)
r = int(np.sum(s > s[0] * 1e-10))
P_row = Vt[:r].T @ Vt[:r]             # projector onto row(X)
for name, l in [("tau alone", np.r_[1, 0, np.zeros(G)]),
                ("educ slope", np.r_[0, 1, np.zeros(G)]),
                ("tau + alpha_1", np.r_[1, 0, 1, np.zeros(G - 1)])]:
    gap = np.linalg.norm(l - P_row @ l)
    ok = "identified" if gap < 1e-8 else "NOT identified"
    print(f"{name:<14s} distance from row(X) = {gap:.3f}  -> {ok}")

# A second (pre-programme) survey round restores identification
y0 = 1.0 + 0.08 * educ + alpha[region] + rng.normal(0, 0.5, n)
Y = np.r_[y0, y]
post = np.r_[np.zeros(n), np.ones(n)]
Xp = np.column_stack([post * np.r_[T, T], post, np.r_[educ, educ],
                      np.vstack([D, D])])
bp = np.linalg.lstsq(Xp, Y, rcond=None)[0]
print(f"\nTwo rounds: columns = {Xp.shape[1]}, rank = "
      f"{np.linalg.matrix_rank(Xp)}, tau-hat (DiD) = {bp[0]:.4f}")
