# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Ramsey toy problem: least-distortion commodity taxes for a revenue target.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy.optimize import brentq, minimize

goods = ["staple food", "fuel", "telecom", "beverages"]
A = np.array([100.0, 60.0, 50.0, 40.0])       # demand scale at price 1
e = np.array([0.3, 0.6, 1.2, 1.8])            # own-price elasticities
R0 = 30.0                                     # revenue target


def demand(t):                                # producer prices equal 1
    return A * (1 + t)**(-e)


def cs_loss(t):                               # loss of consumer surplus
    q = 1 + t
    return np.sum(A * (q**(1 - e) - 1) / (1 - e))


revenue = lambda t: np.sum(t * demand(t))
res = minimize(cs_loss, np.full(4, 0.1), method="SLSQP", tol=1e-13,
               bounds=[(0, None)] * 4,
               constraints=[{"type": "ineq", "fun": lambda t: revenue(t) - R0}])
t = res.x
print("Ramsey taxes for revenue R0 = %.0f (simulated demands)" % R0)
print("  good          elast.   tax rate   t/(1+t)   e * t/(1+t)")
for g, ei, ti in zip(goods, e, t):
    print("  %-12s %6.2f   %8.4f   %7.4f   %9.4f" %
          (g, ei, ti, ti / (1 + ti), ei * ti / (1 + ti)))
print("  inverse elasticity rule: e_i t_i/(1+t_i) is the same for all i")
eb_r = cs_loss(t) - revenue(t)

# uniform tax raising the same revenue
tu = brentq(lambda s: revenue(np.full(4, s)) - R0, 0, 2)
eb_u = cs_loss(np.full(4, tu)) - R0
print("\nexcess burden: Ramsey %.4f   uniform rate %.4f (t = %.4f)" %
      (eb_r, eb_u, tu))

# multiplier = marginal cost of public funds (MCPF)
mu = demand(t) / (demand(t) * (1 - e * t / (1 + t)))   # dLoss/dt / dRev/dt
print("ratio dLoss/dt_i / dRev/dt_i (all equal mu):", np.round(mu, 4))


def ramsey_exact(R):
    """t_i/(1+t_i) = k/e_i  =>  t_i = k/(e_i - k); choose k to hit R."""
    k = brentq(lambda k: revenue(k / (e - k)) - R, 0, 0.99 * e.min())
    return k / (e - k)


h = 1e-4
mcpf = (cs_loss(ramsey_exact(R0 + h)) - cs_loss(ramsey_exact(R0 - h))) / (2 * h)
print("envelope theorem: dLoss/dR = %.4f  (MCPF; excess burden per extra"
      " GHS = %.4f)" % (mcpf, mcpf - 1))
