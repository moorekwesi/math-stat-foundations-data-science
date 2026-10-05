# DEP 803 -- Chapter 7: Estimation Principles
# KL divergence: closed form, local Fisher approximation, and pseudo-true parameters.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats

rng = np.random.default_rng(803)


def kl_normal(m1, s1, m2, s2):
    return np.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5


# Part A: closed form vs numerical integration; asymmetry
f, g = stats.norm(0, 1), stats.norm(1, 2)
num = integrate.quad(lambda x: f.pdf(x) * (f.logpdf(x) - g.logpdf(x)),
                     -np.inf, np.inf)[0]
print("Part A: KL divergences between normals")
print(f"  KL(N(0,1) || N(1,4)) closed form {kl_normal(0, 1, 1, 2):.5f},"
      f" quadrature {num:.5f}")
print(f"  KL(N(1,4) || N(0,1)) closed form {kl_normal(1, 2, 0, 1):.5f}"
      "  (KL is not symmetric)")

# Local approximation KL(p, p + d) ~ d^2 I(p) / 2 for the Bernoulli
p = 0.3
print("  Bernoulli p = 0.3: KL(p || p+d) vs d^2 I(p)/2, I(p) = 1/(p(1-p))")
for d in (0.1, 0.03, 0.01):
    q = p + d
    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    approx = d**2 / (2 * p * (1 - p))
    print(f"    d = {d:5.2f}: KL = {kl:.3e}, approx = {approx:.3e},"
          f" ratio = {kl / approx:.4f}")

# Part B: misspecified exponential model fitted to gamma(3, 1) data
k0 = 3.0
f0 = stats.gamma(k0)
lam_star = 1 / f0.mean()                      # pseudo-true rate
print("\nPart B: exponential model fitted to gamma(shape 3) data")
print(f"  pseudo-true rate 1/E[X] = {lam_star:.4f}")
for n in (50, 500, 5000, 50000):
    x = f0.rvs(size=n, random_state=rng)
    print(f"  n = {n:>6}: MLE rate = {1 / x.mean():.4f}")
lams = np.linspace(0.1, 1.0, 300)
# KL(f0 || Exp(lam)) = -H(f0) - E log f_lam = -H(f0) - log lam + lam E[X]
kl_curve = -f0.entropy() - np.log(lams) + lams * f0.mean()
print(f"  KL(f0 || best exponential) = {kl_curve.min():.4f} nats"
      " (misspecification gap)")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(lams, kl_curve, "k", lw=2, label="KL(f0 || Exp(rate))")
for n, col in ((30, "tab:orange"), (300, "tab:blue")):
    x = f0.rvs(size=n, random_state=rng)
    emp = -np.log(lams) + lams * x.mean() - f0.entropy()
    ax.plot(lams, emp, color=col, lw=1.2,
            label=f"avg neg. log-lik - H(f0), n = {n}")
    ax.plot(1 / x.mean(), emp.min(), "o", color=col)
ax.axvline(lam_star, color="tab:red", ls="--", lw=0.8,
           label="pseudo-true rate")
ax.set_xlabel("exponential rate")
ax.set_ylabel("nats")
ax.set_ylim(0, 1.2)
ax.legend(frameon=False, fontsize=8.5)
plt.savefig("ch07_kl_pseudotrue.pdf", bbox_inches="tight")
