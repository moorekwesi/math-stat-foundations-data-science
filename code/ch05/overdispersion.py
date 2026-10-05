# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Simulated clinic visits: Poisson versus negative binomial (gamma-Poisson mixture)
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 5000                          # simulated households
mu, r = 3.0, 1.5                  # mean visits per year, NB size (shape)
frailty = rng.gamma(shape=r, scale=1 / r, size=n)   # mean 1, var 1/r
visits = rng.poisson(mu * frailty)

m, v = visits.mean(), visits.var(ddof=1)
print(f"sample mean = {m:.3f}, sample variance = {v:.3f}, ratio = {v / m:.2f}")
print(f"theory: Var = mu + mu^2/r = {mu + mu**2 / r:.3f}")

# method-of-moments negative binomial: Var = m + m^2/r  =>  r = m^2/(v - m)
r_hat = m**2 / (v - m)
p_hat = r_hat / (r_hat + m)       # scipy nbinom(r, p) has mean r(1-p)/p
print(f"MoM estimates: r = {r_hat:.3f}, p = {p_hat:.3f}")

k = np.arange(0, 16)
obs = np.array([(visits == j).mean() for j in k])
pois = stats.poisson.pmf(k, m)
nb = stats.nbinom.pmf(k, r_hat, p_hat)
print(" k   observed  Poisson  NegBin")
for j in [0, 1, 2, 5, 10, 15]:
    print(f"{j:2d}{obs[j]:10.4f}{pois[j]:9.4f}{nb[j]:8.4f}")
ll_p = stats.poisson.logpmf(visits, m).sum()
ll_nb = stats.nbinom.logpmf(visits, r_hat, p_hat).sum()
print(f"log-likelihood: Poisson = {ll_p:.1f}, NegBin = {ll_nb:.1f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar(k, obs, color="#bbbbbb", label="simulated data")
ax.plot(k, pois, "o-", color="#1f4e79", label="Poisson fit")
ax.plot(k, nb, "s--", color="#cc9900", label="negative binomial fit")
ax.set_xlabel("clinic visits per household per year")
ax.set_ylabel("proportion")
ax.legend(frameon=False)
plt.savefig("ch05_overdispersion.pdf", bbox_inches="tight")
