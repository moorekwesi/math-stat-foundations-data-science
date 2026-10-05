# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Policy-relevant integrals with closed forms, checked by numerical quadrature.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special, stats

rows = []

# 1. Change in consumer surplus from a price subsidy (linear demand)
a, b, p0, p1 = 100.0, 4.0, 10.0, 8.0
demand = lambda p: a - b * p
exact = a * (p0 - p1) - 0.5 * b * (p0**2 - p1**2)
rows.append(("dCS, linear demand", exact, integrate.quad(demand, p1, p0)[0]))

# 2. Same subsidy, constant-elasticity demand D(p) = A p^(-eps)
A, eps = 100.0, 1.5
exact = A / (1 - eps) * (p0 ** (1 - eps) - p1 ** (1 - eps))
num = integrate.quad(lambda p: A * p ** (-eps), p1, p0)[0]
rows.append(("dCS, CES demand", exact, num))

# 3. Present value of a benefit stream B for T years at rate r
B, r, T = 50.0, 0.08, 20.0
exact = B * (1 - np.exp(-r * T)) / r
rows.append(("PV of benefits", exact,
             integrate.quad(lambda t: B * np.exp(-r * t), 0, T)[0]))

# 4. Expected income when log income ~ N(mu, s^2)
mu, s = 7.0, 0.8
f = lambda y: y * stats.lognorm.pdf(y, s, scale=np.exp(mu))
rows.append(("E[income], lognormal", np.exp(mu + s**2 / 2),
             integrate.quad(f, 0, np.inf)[0]))

# 5. Probit probability P(y=1) = Phi(x'beta) as an integral of the density
xb = 0.4
rows.append(("probit P(y=1)", stats.norm.cdf(xb),
             integrate.quad(stats.norm.pdf, -np.inf, xb)[0]))

# 6. Beta-binomial marginal likelihood: k successes in n trials, flat prior
n, k = 30, 21
exact = special.comb(n, k) * special.beta(k + 1, n - k + 1)
lik = lambda th: special.comb(n, k) * th**k * (1 - th) ** (n - k)
rows.append(("marginal likelihood", exact, integrate.quad(lik, 0, 1)[0]))

# 7. Posterior mean of theta under the same model: (k+1)/(n+2)
num = integrate.quad(lambda th: th * lik(th), 0, 1)[0] / rows[-1][2]
rows.append(("posterior mean", (k + 1) / (n + 2), num))

print(f"{'quantity':24s}{'closed form':>16s}{'quad':>16s}{'abs error':>11s}")
for name, ex, nu in rows:
    print(f"{name:24s}{ex:16.8f}{nu:16.8f}{abs(ex - nu):11.1e}")

# Figure: consumer surplus gain under the linear demand curve
q = np.linspace(0, a, 200)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(q, (a - q) / b, color="navy", lw=2, label="inverse demand P(Q)")
ax.fill_between([0, demand(p0)], p1, p0, color="seagreen", alpha=0.35,
                label=f"gain in CS = {rows[0][1]:.0f}")
qq = np.linspace(demand(p0), demand(p1), 50)
ax.fill_between(qq, p1, (a - qq) / b, color="seagreen", alpha=0.35)
for p in (p0, p1):
    ax.axhline(p, color="gray", ls=":", lw=1)
ax.text(2, p0 + 0.4, "P0 = 10")
ax.text(2, p1 - 1.2, "P1 = 8 (subsidised)")
ax.set_xlabel("quantity Q")
ax.set_ylabel("price P")
ax.set_xlim(0, a)
ax.set_ylim(0, a / b)
ax.legend()
plt.savefig("ch10_consumer_surplus.pdf", bbox_inches="tight")
