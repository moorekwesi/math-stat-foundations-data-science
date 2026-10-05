# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Rank, the dummy variable trap and numerical rank in a simulated survey
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, G = 800, 16                       # households, regions (Ghana has 16)
region = rng.integers(0, G, size=n)
educ = rng.normal(9, 3, size=n)      # years of schooling (simulated)
D = np.eye(G)[region]                # n x G matrix of region dummies
one = np.ones((n, 1))


def report(name, X):
    r = np.linalg.matrix_rank(X)
    k = X.shape[1]
    nullity = k - r
    flag = "full column rank" if nullity == 0 else f"RANK DEFICIENT by {nullity}"
    print(f"{name:<34s} k = {k:2d}  rank = {r:2d}  -> {flag}")


report("const + educ", np.column_stack([one, educ]))
report("const + educ + all 16 dummies", np.column_stack([one, educ, D]))
report("const + educ + 15 dummies", np.column_stack([one, educ, D[:, 1:]]))
report("no const + educ + 16 dummies", np.column_stack([educ, D]))

# The null vector of the trapped design: (1, 0, -1, ..., -1)
X_trap = np.column_stack([one, educ, D])
c = np.r_[1.0, 0.0, -np.ones(G)]
print(f"\n||X_trap c|| = {np.linalg.norm(X_trap @ c):.1e}  (exact dependence)")

# Rank-nullity check: rank + dim null = number of columns
_, s, Vt = np.linalg.svd(X_trap)
tol = s.max() * max(X_trap.shape) * np.finfo(float).eps
null_dim = int(np.sum(s <= tol))
print(f"rank + nullity = {np.linalg.matrix_rank(X_trap)} + {null_dim} = "
      f"{X_trap.shape[1]}")

# Near collinearity: numerical rank depends on the tolerance
print("\nNear-collinear regressor  x3 = x1 + x2 + delta * noise")
x1, x2 = rng.normal(size=n), rng.normal(size=n)
for delta in [1e-2, 1e-8, 1e-13, 1e-16]:
    X = np.column_stack([one, x1, x2, x1 + x2 + delta * rng.normal(size=n)])
    s = np.linalg.svd(X, compute_uv=False)
    r_def = np.linalg.matrix_rank(X)
    r_loose = np.linalg.matrix_rank(X, tol=1e-6 * s[0])
    print(f"delta = {delta:7.0e}  s_min/s_max = {s[-1] / s[0]:9.2e}  "
          f"rank(default) = {r_def}  rank(tol=1e-6) = {r_loose}")
