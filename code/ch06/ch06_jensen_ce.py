# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Jensen's inequality, certainty equivalents and the value of insurance (CRRA).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)


def u(c, g):
    """CRRA utility with relative risk aversion g."""
    return np.log(c) if g == 1 else c**(1 - g) / (1 - g)


def u_inv(v, g):
    return np.exp(v) if g == 1 else ((1 - g) * v)**(1 / (1 - g))


# 1. Lognormal consumption: CE has the closed form exp(mu + (1-g) s^2 / 2)
mu, s = 7.0, 0.6
c = np.exp(rng.normal(mu, s, size=1_000_000))
print("Lognormal consumption: E[C] =", f"{c.mean():.1f}")
print(f"{'gamma':>6}{'CE (MC)':>10}{'CE exact':>10}{'premium':>9}{'% of E[C]':>11}")
for g in [0.5, 1, 2, 3]:
    ce = u_inv(np.mean(u(c, g)), g)
    exact = np.exp(mu + (1 - g) * s**2 / 2)
    rp = c.mean() - ce
    print(f"{g:6.1f}{ce:10.1f}{exact:10.1f}{rp:9.1f}{100 * rp / c.mean():10.1f}%")

# 2. A farm household: income 1000 in a normal year, 400 in a drought (prob 0.2)
w = np.array([1000.0, 400.0])
pr = np.array([0.8, 0.2])
g = 2
Ew = pr @ w
ce = u_inv(pr @ u(w, g), g)
print(f"\nFarm income: E[W] = {Ew:.1f}, CE (gamma = 2) = {ce:.1f}")
print(f"  risk premium                   = {Ew - ce:.1f}")
print(f"  actuarially fair premium       = {w[0] - Ew:.1f}")
print(f"  max premium for full insurance = {w[0] - ce:.1f}")

# Figure: log utility, chord, E[u(W)] < u(E[W]) and the certainty equivalent
cc = np.linspace(300, 1100, 300)
ul = np.log(w)
Eu = pr @ ul
ce1 = np.exp(Eu)
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(cc, np.log(cc), color="navy", lw=2, label="u(w) = log w")
ax.plot(w, ul, "o--", color="gray", label="chord")
ax.plot(Ew, Eu, "s", color="firebrick", label="E[u(W)]")
ax.plot(Ew, np.log(Ew), "^", color="darkgreen", label="u(E[W])")
ax.hlines(Eu, 300, Ew, color="firebrick", lw=0.6, ls=":")
ax.vlines([ce1, Ew], 5.7, [Eu, np.log(Ew)], colors="k", lw=0.6, ls=":")
ax.text(ce1 - 75, 5.73, "CE")
ax.text(Ew + 10, 5.73, "E[W]")
ax.set_xlabel("income w")
ax.set_ylabel("utility")
ax.set_ylim(5.7, 7.05)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Jensen: E[u(W)] <= u(E[W]) for concave u")
plt.savefig("ch06_jensen.pdf", bbox_inches="tight")
print(f"\nlog utility: CE = {ce1:.1f}, premium = {Ew - ce1:.1f}")
