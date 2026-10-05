# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Social planner: transfers with equity weights and inequality aversion.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy.optimize import brentq, minimize

# Five simulated regional groups: pre-transfer income (GHS per capita/month)
groups = ["North", "Upper", "Middle", "Coast", "Capital"]
y = np.array([320.0, 380.0, 520.0, 650.0, 900.0])
pop = np.array([1.2, 0.9, 2.0, 1.5, 2.4])          # millions of people
Bud = 300.0                                        # GHS million per month


def uprime(c, eta):
    return c**(-eta)


def plan(w, eta):
    """max sum pop_i w_i u(y_i + T_i)  s.t.  sum pop_i T_i = Bud, T >= 0.
    KKT: w_i u'(y_i + T_i) = lam if T_i > 0 (fill from the bottom)."""
    def spend(lam):
        c = (lam / w)**(-1 / eta)                  # inverse of w u'(c) = lam
        return np.sum(pop * np.maximum(c - y, 0)) - Bud
    lam = brentq(spend, 1e-12, np.max(w * uprime(y, eta)))
    T = np.maximum((lam / w)**(-1 / eta) - y, 0)
    return T, lam


print("Budget GHS %.0f million; transfers T_i in GHS per capita" % Bud)
print("  weights      eta   " + "  ".join("%7s" % g for g in groups))
for wname, w in [("utilitarian", np.ones(5)),
                 ("equity", np.array([1.5, 1.3, 1.0, 1.0, 0.8]))]:
    for eta in [0.5, 1.0, 2.0]:
        T, lam = plan(w, eta)
        print("  %-11s %4.1f   " % (wname, eta) +
              "  ".join("%7.1f" % t for t in T))

# numerical check of one case with SLSQP (log utility, equity weights)
w = np.array([1.5, 1.3, 1.0, 1.0, 0.8])
T, lam = plan(w, 1.0)
W = lambda t: -np.sum(pop * w * np.log(y + t))
r = minimize(W, np.full(5, Bud / pop.sum()), method="SLSQP", tol=1e-12,
             bounds=[(0, None)] * 5,
             constraints=[{"type": "eq", "fun": lambda t: pop @ t - Bud}])
print("\nSLSQP check (equity, eta=1):", np.round(r.x, 1))
print("multiplier lambda = %.6f = marginal social welfare of GHS 1 million"
      % lam)
print("weighted marginal utilities w_i/(y_i+T_i):",
      np.round(w / (y + T), 6))
