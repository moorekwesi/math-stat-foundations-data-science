# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Least-squares balancing of (illustrative) national accounts with pinv
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

names = ["VA agric", "VA indus", "VA serv", "C house", "G govt", "I invest",
         "NX net ex", "COMP empl", "GOS surpl"]
x0 = np.array([210., 330., 480., 700., 150., 230., -40., 520., 470.])
cv = np.array([.08, .04, .04, .03, .01, .10, .15, .05, .08])  # rel. std errors
V = np.diag((cv * np.abs(x0)) ** 2)        # prior variances of the estimates
prod = np.r_[1, 1, 1, 0, 0, 0, 0, 0, 0]    # production approach
expd = np.r_[0, 0, 0, 1, 1, 1, 1, 0, 0]    # expenditure approach
inc = np.r_[0, 0, 0, 0, 0, 0, 0, 1, 1]     # income approach
A = np.vstack([prod - expd, prod - inc, expd - inc])   # A x = 0 (redundant!)
print(f"constraints: {A.shape[0]}, rank(A) = {np.linalg.matrix_rank(A)}")
print("GDP before:  production {:.1f}, expenditure {:.1f}, income {:.1f}"
      .format(prod @ x0, expd @ x0, inc @ x0))

S = A @ V @ A.T                            # singular because rank(A) = 2
print(f"cond(A V A^T) = {np.linalg.cond(S):.1e}")
x_star = x0 - V @ A.T @ np.linalg.pinv(S) @ (A @ x0)
print("GDP after:   production {:.1f}, expenditure {:.1f}, income {:.1f}"
      .format(prod @ x_star, expd @ x_star, inc @ x_star))
print(f"max |A x*| = {np.abs(A @ x_star).max():.1e}")

# Same answer using only the two independent constraints
A2 = A[:2]
x2 = x0 - V @ A2.T @ np.linalg.solve(A2 @ V @ A2.T, A2 @ x0)
print(f"max |x* - x*(2 constraints)| = {np.abs(x_star - x2).max():.1e}\n")
print(f"{'item':<10s} {'prelim':>8s} {'balanced':>9s} {'change':>8s} {'cv':>5s}")
for nm, a, b, c in zip(names, x0, x_star, cv):
    print(f"{nm:<10s} {a:8.1f} {b:9.1f} {b - a:8.1f} {c:5.2f}")
