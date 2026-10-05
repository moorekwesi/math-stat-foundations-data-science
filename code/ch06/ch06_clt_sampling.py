# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Sampling distribution of standardised means; KS distance and Berry-Esseen bound.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
reps = 50_000
C_BE = 0.4748                          # Shevtsova (2011) constant, iid case


def ks_to_normal(z):
    return stats.kstest(z, "norm").statistic


# Exponential(1): mu = sd = 1, rho = E|X - mu|^3 = 12/e - 2
rho_exp = 12 / np.e - 2
# Bernoulli(0.1): rho = p q (p^2 + q^2), sd = sqrt(p q)
p = 0.1
q = 1 - p
rho_ber, sd_ber = p * q * (p**2 + q**2), np.sqrt(p * q)
print(f"{'n':>5}{'KS exp':>9}{'BE exp':>9}{'KS bern':>9}{'BE bern':>9}"
      f"{'skew exp':>10}")
Z = {}
for n in [5, 30, 300]:
    xbar = rng.gamma(n, 1.0, size=reps) / n                  # Exponential mean
    z = np.sqrt(n) * (xbar - 1.0)
    Z[n] = z
    zb = (rng.binomial(n, p, size=reps) / n - p) / (sd_ber / np.sqrt(n))
    be_exp = C_BE * rho_exp / np.sqrt(n)
    be_ber = C_BE * rho_ber / (sd_ber**3 * np.sqrt(n))
    print(f"{n:5d}{ks_to_normal(z):9.4f}{be_exp:9.4f}{ks_to_normal(zb):9.4f}"
          f"{be_ber:9.4f}{stats.skew(z):10.3f}")
print("(skewness of the mean is 2/sqrt(n): "
      + ", ".join(f"{2 / np.sqrt(n):.3f}" for n in [5, 30, 300]) + ")")

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
t = np.linspace(-4, 4, 300)
for ax, n in zip(axes, Z):
    ax.hist(Z[n], bins=80, range=(-4, 4), density=True, color="lightsteelblue")
    ax.plot(t, stats.norm.pdf(t), color="firebrick", lw=1.5)
    ax.set_title(f"n = {n}", fontsize=10)
    ax.set_xlabel("sqrt(n)(Xbar - mu)/sd")
axes[0].set_ylabel("density")
fig.tight_layout()
plt.savefig("ch06_clt.pdf", bbox_inches="tight")
