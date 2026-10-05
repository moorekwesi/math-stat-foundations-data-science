# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Vectorised computation and NumPy broadcasting rules, with a classic pitfall.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
X = rng.normal(loc=[10, 50, 3], scale=[2, 15, 1], size=(5, 3)).round(1)
print("X (5 x 3) =\n", X)
mu = X.mean(axis=0)                       # shape (3,)
Xc = X - mu                               # (5,3) - (3,) -> row-wise subtraction
print("column means after centring:", np.round(Xc.mean(axis=0), 12) + 0.0)
w = np.array([0.1, 0.2, 0.3, 0.2, 0.2])   # one weight per row
print("weighted column means w^T X :", np.round(w @ X, 3))
Xw = X * w[:, None]                       # (5,3) * (5,1) scales each row
print("row scaling equals diag(w) X:", np.allclose(Xw, np.diag(w) @ X))

# outer product and pairwise distances without loops
a, b = np.arange(1, 4), np.arange(1, 5)
print("outer product a[:,None]*b[None,:] =\n", a[:, None] * b[None, :])
diff = X[:, None, :] - X[None, :, :]      # shape (5, 5, 3)
Dist = np.sqrt((diff**2).sum(axis=2))
print("pairwise Euclidean distances (rounded):\n", Dist.round(1))

# pitfall: a column vector (n,1) minus a flat vector (n,) gives an (n,n) matrix
y = rng.standard_normal(5)
yhat = y.reshape(-1, 1)                   # e.g. a prediction returned as (5,1)
resid = y - yhat
print("shape of y - yhat when yhat is (5,1):", resid.shape, " <- silent bug!")
print("correct: y - yhat.ravel() has shape", (y - yhat.ravel()).shape)
try:
    X + np.ones(5)                        # (5,3) + (5,) is incompatible
except ValueError as err:
    print("ValueError:", err)
