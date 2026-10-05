# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Centring matrix, sample covariance and correlation from a data matrix.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import pandas as pd

rng = np.random.default_rng(803)
n = 500
# simulated household survey (illustrative only)
school = rng.integers(0, 19, n).astype(float)
hhsize = rng.poisson(4.5, n) + 1.0
dist = rng.gamma(2.0, 4.0, n)                        # km to nearest market
loginc = 1.2 + 0.09 * school - 0.04 * dist + 0.05 * hhsize \
    + rng.normal(0, 0.45, n)
df = pd.DataFrame({"log_income": loginc, "schooling": school,
                   "hh_size": hhsize, "dist_market": dist})
print(df.describe().loc[["mean", "std"]].round(3))
X = df.to_numpy()                                    # n x p float64 array
print("X:", X.shape, X.dtype)

one = np.ones((n, 1))
H = np.eye(n) - one @ one.T / n                      # centring matrix
print("H symmetric:", np.allclose(H, H.T), " H idempotent:", np.allclose(H @ H, H),
      " trace(H) =", round(np.trace(H), 8))
print("H X equals X - column means:", np.allclose(H @ X, X - X.mean(axis=0)))

S = X.T @ H @ X / (n - 1)                            # sample covariance
Xc = X - X.mean(axis=0)
S_fast = Xc.T @ Xc / (n - 1)                         # same, without the n x n H
print("S matches np.cov:", np.allclose(S, np.cov(X, rowvar=False)),
      "  matches fast version:", np.allclose(S, S_fast))
d = np.sqrt(np.diag(S))
R = S / np.outer(d, d)                               # D^-1/2 S D^-1/2
Z = Xc / d                                           # standardised data
print("R matches np.corrcoef:", np.allclose(R, np.corrcoef(X, rowvar=False)),
      "  Z^T Z/(n-1) = R:", np.allclose(Z.T @ Z / (n - 1), R))
print("correlation matrix R:")
print(pd.DataFrame(R, index=df.columns, columns=df.columns).round(3))
print("eigenvalues of S (all >= 0):", np.linalg.eigvalsh(S).round(4))
print("memory of H: %.1f MB versus X: %.3f MB" % (H.nbytes / 1e6, X.nbytes / 1e6))

