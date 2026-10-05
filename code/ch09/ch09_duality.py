# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Lagrange dual functions: strong duality (convex QP) and a gap (knapsack).
# (c) 2026 Stephen E. Moore. MIT Licence.
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ---- (1) convex QP: min x1^2 + x2^2  s.t.  x1 + x2 >= 1 ------------------
# L(x, lam) = x'x + lam (1 - x1 - x2); minimiser x = lam/2 (1, 1)
q = lambda lam: lam - lam**2 / 2                     # dual function
lams = np.linspace(0, 2.5, 300)
p_star, d_star = 0.5, q(1.0)
print("Convex QP:  p* = %.4f,  d* = max q(lam) = %.4f at lam = 1" %
      (p_star, d_star))
print("  duality gap = %.2e  (Slater holds: x = (1, 1) is strictly feasible)"
      % (p_star - d_star))

# ---- (2) 0-1 project selection (knapsack): duality gap ------------------
v = np.array([10.0, 13.0, 7.0, 8.0, 5.0, 6.0])      # benefit (simulated)
c = np.array([5.0, 7.0, 4.0, 5.0, 3.0, 4.0])        # cost, GHS million
B = 13.0
best, best_x = -np.inf, None
for bits in itertools.product([0, 1], repeat=len(v)):
    xb = np.array(bits)
    if c @ xb <= B and v @ xb > best:
        best, best_x = v @ xb, xb


def knap_dual(lam):
    """q(lam) = max over x in {0,1}^n of v'x - lam (c'x - B)."""
    return lam * B + np.maximum(v - lam * c, 0).sum()


lam_k = np.linspace(0, 3, 600)
qk = np.array([knap_dual(L) for L in lam_k])
cands = np.r_[0, v / c]                              # kinks of q(lam)
qc = np.array([knap_dual(L) for L in cands])
d_k, lam_best = qc.min(), cands[np.argmin(qc)]
print("\nKnapsack: projects chosen", best_x, " cost = %.0f  p* = %.1f" %
      (c @ best_x, best))
print("  benefit-cost ratios:", np.round(v / c, 3))
print("  dual bound d* = min q(lam) = %.3f at lam = %.3f" % (d_k, lam_best))
print("  duality gap d* - p* = %.3f  (weak duality: d* >= p*)" % (d_k - best))

# ---- figure -------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
axs[0].plot(lams, q(lams), lw=2, label="dual function q(lam)")
axs[0].axhline(p_star, ls="--", color="C3", label="primal optimum p* = 0.5")
axs[0].plot(1, d_star, "o", color="C3")
axs[0].set_ylim(-0.6, 0.8)
axs[0].set_xlabel("multiplier lam")
axs[0].set_title("Convex QP: zero duality gap", fontsize=10)
axs[1].plot(lam_k, qk, lw=2, label="dual function q(lam)")
axs[1].axhline(best, ls="--", color="C3",
               label="integer optimum p* = %.0f" % best)
axs[1].plot(lam_best, d_k, "o", color="C0")
axs[1].set_ylim(20, 45)
axs[1].set_xlabel("multiplier lam")
axs[1].set_title("0-1 project selection: gap = %.2f" % (d_k - best),
                 fontsize=10)
for ax in axs:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch09_duality.pdf", bbox_inches="tight")
