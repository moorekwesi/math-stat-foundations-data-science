# DEP 803 -- Chapter 0: Mathematical Refresher
# Sigma and product notation, double sums, weighted averages, mean and variance.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

# sum_{i=1}^{4} (2i - 1)  and  prod_{k=1}^{3} (1 + k/2)
i = np.arange(1, 5)                 # 1, 2, 3, 4  (note: stop value 5 excluded)
print("sum_{i=1}^4 (2i-1)      =", np.sum(2 * i - 1))
k = np.arange(1, 4)
print("prod_{k=1}^3 (1 + k/2)  =", np.prod(1 + k / 2))

# double sum sum_{i=1}^3 sum_{j=1}^2 i*j as a table, then summed
a = np.arange(1, 4)[:, None] * np.arange(1, 3)[None, :]
print("table of i*j (rows i, columns j):")
print(a)
print("double sum              =", a.sum(), "= (1+2+3)*(1+2) =", 6 * 3)

# mean and variance written with sigma
x = np.array([2.0, 4.0, 6.0, 8.0])
n = x.size
xbar = x.sum() / n
ss = np.sum((x - xbar) ** 2)
print("x =", x, " mean =", xbar)
print("deviations x_i - xbar   =", x - xbar, " (they sum to", (x - xbar).sum(), ")")
print("sum of squared deviations =", ss)
print("sample variance s^2     =", round(ss / (n - 1), 4),
      " numpy:", round(np.var(x, ddof=1), 4))

# weighted average: portfolio return and a trial-balance check
w = np.array([0.5, 0.3, 0.2])       # portfolio weights, sum to 1
r = np.array([0.12, 0.08, -0.04])   # asset returns
print("weights sum to", w.sum(), "; portfolio return sum w_i r_i =",
      round(np.sum(w * r), 4))
debits = np.array([12_000, 3_500, 800, 4_700])
credits = np.array([15_000, 6_000])
print("trial balance: total debits =", debits.sum(),
      " total credits =", credits.sum())
