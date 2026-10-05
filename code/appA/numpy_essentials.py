# DEP 803 -- Appendix A: Python Toolkit
# NumPy essentials: arrays, broadcasting, linear algebra and random numbers.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)            # reproducible random numbers

# Arrays, shapes and broadcasting
X = rng.normal(size=(5, 3))                 # 5 observations, 3 variables
mu = X.mean(axis=0)                         # column means, shape (3,)
Xc = X - mu                                 # broadcasting subtracts mu from each row
print("shape of X:", X.shape, " column means after centring:",
      np.abs(np.round(Xc.mean(axis=0), 12)))

# Matrix products and solving linear systems
S = Xc.T @ Xc / (X.shape[0] - 1)            # sample covariance matrix
print("covariance matrix equals np.cov:", np.allclose(S, np.cov(X, rowvar=False)))
b = np.array([1.0, 2.0, 3.0])
x = np.linalg.solve(S, b)                   # never form inv(S) @ b
print("residual of S x = b:", np.linalg.norm(S @ x - b))

# Decompositions
lam, V = np.linalg.eigh(S)                  # symmetric eigen-decomposition
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print("eigenvalues of S:      ", np.round(lam[::-1], 4))
print("s**2/(n-1) from SVD:   ", np.round(s**2 / (X.shape[0] - 1), 4))

# Least squares with an intercept
n = 200
x1 = rng.uniform(0, 10, n)
y = 2.0 + 0.5 * x1 + rng.normal(0, 1, n)
A = np.column_stack([np.ones(n), x1])
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
print("OLS coefficients (true 2.0, 0.5):", np.round(beta, 3))

# Boolean indexing and vectorised summaries
income = rng.lognormal(mean=8.0, sigma=0.8, size=10_000)
z = 1500.0                                  # an illustrative poverty line
print(f"headcount ratio below z: {np.mean(income < z):.3f}")
print(f"mean poverty gap ratio:  {np.mean(np.maximum(z - income, 0) / z):.3f}")
