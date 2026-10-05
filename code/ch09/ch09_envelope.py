# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Multipliers as shadow prices: envelope theorem and bordered Hessian.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Simulated district programme: outcome f_i(x) = a_i log(1 + x_i / b_i)
names = ["education", "health", "agric. ext."]
a = np.array([3.0, 4.0, 2.0])                 # scale of returns (simulated)
b = np.array([1.0, 2.0, 0.5])                 # saturation (GHS million)


def allocate(B):
    """KKT solution x_i = max(0, a_i/lam - b_i), lam found by bisection."""
    lam = brentq(lambda L: np.maximum(a / L - b, 0).sum() - B, 1e-8, 1e3,
                 xtol=1e-14)
    return np.maximum(a / lam - b, 0), lam


def V(B):
    return np.sum(a * np.log1p(allocate(B)[0] / b))


B0 = 6.0
x0, lam0 = allocate(B0)
h = 1e-4
dV = (V(B0 + h) - V(B0 - h)) / (2 * h)
print("Budget B = %.1f (GHS million), simulated returns" % B0)
for n, xi, mi in zip(names, x0, a / (b + x0)):
    print("  %-12s x = %6.3f   marginal return = %.4f" % (n, xi, mi))
print("Lagrange multiplier lambda       = %.6f" % lam0)
print("Finite-difference dV/dB          = %.6f" % dV)

# comparative statics: dx/dB (all programmes interior here)
dx = (allocate(B0 + h)[0] - allocate(B0 - h)[0]) / (2 * h)
print("dx/dB (numerical)                =", np.round(dx, 4),
      " sum = %.4f" % dx.sum())
print("closed form a_i / sum(a)         =", np.round(a / a.sum(), 4))
print("dlambda/dB: numerical %.5f, closed form -lam^2/sum(a) = %.5f" %
      ((allocate(B0 + h)[1] - allocate(B0 - h)[1]) / (2 * h),
       -lam0**2 / a.sum()))

# ---- bordered Hessian second-order check ------------------------------
H = np.diag(-a / (b + x0)**2)                 # Hessian of Lagrangian in x
g = np.ones(3)                                # gradient of the constraint
Hb = np.block([[np.zeros((1, 1)), g[None, :]], [g[:, None], H]])
minors = [np.linalg.det(Hb[:r, :r]) for r in (3, 4)]
print("\nBordered Hessian leading minors of order 3, 4:", np.round(minors, 4))
print("signs (+, -) as required for a constrained maximum:",
      bool(minors[0] > 0 and minors[1] < 0))

# ---- figure: value function and tangent with slope lambda --------------
Bs = np.linspace(0.05, 14, 200)
Vs = np.array([V(B) for B in Bs])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(Bs, Vs, lw=2, label="value function V(B)")
ax.plot(Bs, V(B0) + lam0 * (Bs - B0), "--", color="C3",
        label="tangent at B0 = 6, slope lambda = %.3f" % lam0)
ax.plot(B0, V(B0), "o", color="C3")
ax.set_xlabel("district budget B (GHS million)")
ax.set_ylabel("total outcome index V(B)")
ax.set_title("Envelope theorem: dV/dB equals the multiplier", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("ch09_envelope.pdf", bbox_inches="tight")
