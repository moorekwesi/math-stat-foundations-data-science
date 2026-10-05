# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Index, summation and product notation translated into NumPy.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, p = 6, 3
# simulated household survey: income (GHS thousand), household size, schooling
X = np.column_stack([rng.lognormal(2.5, 0.6, n).round(1),
                     rng.integers(1, 9, n),
                     rng.integers(0, 17, n)]).astype(float)
print("Data matrix X has shape (n, p) =", X.shape)
print(X)
print("x_{31} (row 3, column 1) = X[2, 0] =", X[2, 0])   # Python is 0-based
print("row x_3      =", X[2])
print("column x^(2) =", X[:, 1])

# summation notation
print("sum_i x_i1                 =", round(X[:, 0].sum(), 2))
print("column means (xbar_j)      =", X.mean(axis=0).round(3))
print("sum_i sum_j x_ij           =", round(X.sum(), 2))
print("sum_i x_i1 x_i3 (= inner)  =", round(X[:, 0] @ X[:, 2], 2))

# product notation: cumulative price level from monthly inflation
pi = np.array([0.021, 0.018, 0.025, 0.030, 0.012, 0.015])
gross = 1 + pi
print("prod_t (1 + pi_t)          =", gross.prod().round(6))
print("exp(sum_t log(1 + pi_t))   =", np.exp(np.log(gross).sum()).round(6))
print("geometric-mean monthly rate=", (gross.prod() ** (1 / pi.size) - 1).round(6))

# conventions for empty sums and products
print("empty sum =", np.sum([]), "   empty product =", np.prod([]))

# order of magnitude: a likelihood is a product of many small numbers
probs = rng.uniform(0.05, 0.6, size=2000)
print("prod of 2000 probabilities =", np.prod(probs))
print("sum of their logs          =", round(np.log(probs).sum(), 3))
print("log10 of the product       =", round(np.log10(probs).sum(), 1))
