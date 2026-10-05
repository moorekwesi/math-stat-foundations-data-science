# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Marshallian demand (Cobb-Douglas, CES): closed form vs scipy.optimize.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy.optimize import minimize, root

p = np.array([2.0, 5.0, 4.0])                 # prices (GHS per unit)
m = 120.0                                     # household budget (GHS)
alpha = np.array([0.3, 0.5, 0.2])             # Cobb-Douglas shares (sum to 1)

# --- 1. Cobb-Douglas: closed form x_i = alpha_i m / p_i ----------------
x_cd = alpha * m / p
lam_cd = 1 / m                                # for u = sum alpha_i log x_i

# --- 2. Direct numerical maximisation with SLSQP -----------------------
negu = lambda x: -np.sum(alpha * np.log(x))
budget = {"type": "eq", "fun": lambda x: m - p @ x, "jac": lambda x: -p}
res = minimize(negu, x0=np.full(3, 5.0), method="SLSQP", constraints=[budget],
               bounds=[(1e-6, None)] * 3, tol=1e-12)


# --- 3. Solve the Lagrange first-order system F(x, lambda) = 0 ---------
def foc(z):
    x, lam = z[:3], z[3]
    return np.r_[alpha / x - lam * p, m - p @ x]


sol = root(foc, x0=np.r_[np.full(3, 5.0), 0.1])
print("Cobb-Douglas, p =", p, " m =", m)
print("  closed form  x =", np.round(x_cd, 6), " lambda =", round(lam_cd, 6))
print("  SLSQP        x =", np.round(res.x, 6))
print("  root(FOC)    x =", np.round(sol.x[:3], 6), " lambda =",
      round(sol.x[3], 6))
print("  budget shares p_i x_i / m =", np.round(p * x_cd / m, 3))

# --- 4. CES utility u = (sum a_i x_i^r)^(1/r), sigma = 1/(1-r) ---------
a = np.array([0.5, 0.3, 0.2])


def ces_demand(p, m, sigma):
    w = a**sigma * p**(1 - sigma)
    return m * a**sigma * p**(-sigma) / w.sum()


print("\nCES demand (closed form vs SLSQP):")
print("  sigma    x1(cf)   x1(num)   x2(cf)   x3(cf)   max|diff|")
for sigma in [0.5, 0.8, 2.0, 5.0]:
    r = 1 - 1 / sigma
    negces = lambda x: -np.log(np.sum(a * x**r)) / r
    out = minimize(negces, x0=np.full(3, 5.0), method="SLSQP",
                   constraints=[budget], bounds=[(1e-8, None)] * 3, tol=1e-12)
    xc = ces_demand(p, m, sigma)
    print("  %5.2f  %8.4f  %8.4f  %8.4f  %8.4f   %.1e" %
          (sigma, xc[0], out.x[0], xc[1], xc[2], np.max(np.abs(xc - out.x))))

# own-price elasticity of good 1 demand (numerical, CES sigma = 2)
h = 1e-5
dp = np.array([h, 0, 0])
e11 = (np.log(ces_demand(p + dp, m, 2.0)[0]) -
       np.log(ces_demand(p - dp, m, 2.0)[0])) / (np.log(p[0] + h) -
                                                np.log(p[0] - h))
s1 = (a**2 * p**(-1))[0] / np.sum(a**2 * p**(-1))   # expenditure share
print("\nCES sigma=2: own-price elasticity of good 1 = %.4f" % e11)
print("theory -sigma + (sigma-1) s1                = %.4f" % (-2 + s1))
