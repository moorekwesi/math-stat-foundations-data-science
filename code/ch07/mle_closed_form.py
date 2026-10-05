# DEP 803 -- Chapter 7: Estimation Principles
# Closed-form MLEs checked against numerical maximisation; the invariance property.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(803)
n = 250
takeup = rng.binomial(1, 0.35, size=n)          # programme take-up (0/1)
visits = rng.poisson(2.4, size=n)               # clinic visits per household
income = rng.normal(5.0, 1.2, size=n)           # log income
wait = rng.exponential(1 / 0.5, size=n)         # waiting time, rate 0.5


def num_mle(negll, x0, bounds):
    return optimize.minimize(negll, x0, bounds=bounds, method="L-BFGS-B").x


norm_fit = num_mle(lambda t: -stats.norm.logpdf(income, t[0], t[1]).sum(),
                   [0.0, 1.0], [(None, None), (1e-6, None)])
fits = [
    ("Bernoulli p", takeup.mean(),
     num_mle(lambda t: -stats.bernoulli.logpmf(takeup, t[0]).sum(), [0.5],
             [(1e-6, 1 - 1e-6)])[0]),
    ("Poisson lambda", visits.mean(),
     num_mle(lambda t: -stats.poisson.logpmf(visits, t[0]).sum(), [1.0],
             [(1e-6, None)])[0]),
    ("Normal mu", income.mean(), norm_fit[0]),
    ("Normal sigma", income.std(ddof=0), norm_fit[1]),
    ("Exponential rate", 1 / wait.mean(),
     num_mle(lambda t: -stats.expon.logpdf(wait, scale=1 / t[0]).sum(),
             [1.0], [(1e-6, None)])[0]),
]
print(f"{'parameter':<17} {'closed form':>12} {'numerical':>12} {'diff':>10}")
for name, cf, nu in fits:
    print(f"{name:<17} {cf:12.6f} {nu:12.6f} {cf - nu:10.2e}")

# Invariance: the MLE of g(theta) is g(theta_hat)
lam = visits.mean()
rate = 1 / wait.mean()
p = takeup.mean()
print("\nInvariance of the MLE")
print(f"  P(no visits)  : exp(-lambda_hat) = {np.exp(-lam):.4f}"
      f"   (sample share of zeros = {np.mean(visits == 0):.4f})")
print(f"  median wait   : log(2)/rate_hat  = {np.log(2) / rate:.4f}"
      f"   (sample median = {np.median(wait):.4f})")
print(f"  odds of uptake: p/(1-p)          = {p / (1 - p):.4f}")
