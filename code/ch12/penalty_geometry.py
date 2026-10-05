# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Geometry of constrained least squares: l1 ball gives corners, l2 ball does not.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

b_ols = np.array([1.6, 0.6])                   # unconstrained minimiser
Q = np.array([[1.0, 0.6], [0.6, 1.0]])         # X'X / n (correlated)
rss = lambda b: (b - b_ols) @ Q @ (b - b_ols)
t = 1.0                                        # budget: ||b|| <= t

# l1 problem with b = u - v, u, v >= 0 (smooth constraints for SLSQP)
obj = lambda w: rss(w[:2] - w[2:])
cons_l1 = {"type": "ineq", "fun": lambda w: t - np.sum(w)}
w = minimize(obj, np.full(4, 0.1), constraints=[cons_l1], method="SLSQP",
             bounds=[(0, None)] * 4, options={"ftol": 1e-14}).x
b_l1 = np.round(w[:2] - w[2:], 6) + 0.0
cons_l2 = {"type": "ineq", "fun": lambda b: t**2 - b @ b}
b_l2 = minimize(rss, [0.3, 0.3], constraints=[cons_l2], method="SLSQP",
                options={"ftol": 1e-14}).x
print("OLS            :", b_ols)
print("l1-constrained :", np.round(b_l1, 4), "<- second coefficient exactly 0")
print("l2-constrained :", np.round(b_l2, 4), "<- both shrunk, none zero")

g = np.linspace(-1.5, 2.5, 300)
B1, B2 = np.meshgrid(g, g)
D1, D2 = B1 - b_ols[0], B2 - b_ols[1]
Z = Q[0, 0] * D1**2 + 2 * Q[0, 1] * D1 * D2 + Q[1, 1] * D2**2
fig, ax = plt.subplots(1, 2, figsize=(8, 3.8))
for a, sol, title in zip(ax, [b_l1, b_l2], [r"lasso ($\ell_1$ ball)",
                                             r"ridge ($\ell_2$ ball)"]):
    a.contour(B1, B2, Z, levels=sorted([rss(sol), 0.05, 1.5, 3.0]), colors="gray",
              linewidths=0.8)
    if "1" in title:
        a.fill([t, 0, -t, 0], [0, t, 0, -t], color="C0", alpha=0.3)
    else:
        a.add_patch(plt.Circle((0, 0), t, color="C1", alpha=0.3))
    a.plot(*b_ols, "k*", ms=10)
    a.plot(*sol, "ro", ms=6)
    a.axhline(0, color="k", lw=0.5)
    a.axvline(0, color="k", lw=0.5)
    a.set_aspect("equal")
    a.set_xlim(-1.5, 2.5)
    a.set_ylim(-1.5, 2.0)
    a.set_title(title)
    a.set_xlabel(r"$\beta_1$")
ax[0].set_ylabel(r"$\beta_2$")
plt.tight_layout()
plt.savefig("ch12_penalty_geometry.pdf", bbox_inches="tight")
