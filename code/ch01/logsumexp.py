# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Stable softmax, log-sum-exp and logit log-likelihoods.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy.special import logsumexp, expit

# utilities of 4 transport modes for one traveller (simulated, large scale)
v = np.array([800.0, 805.0, 790.0, 802.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(v) / np.exp(v).sum()
print("naive softmax  :", naive)


def softmax(v):
    """Shift by the maximum before exponentiating: exact and safe."""
    z = np.exp(v - v.max())
    return z / z.sum()


def lse(v):
    m = v.max()
    return m + np.log(np.exp(v - m).sum())


print("stable softmax :", softmax(v).round(6))
print("sum of probs   :", softmax(v).sum())
print("log-sum-exp    : ours = %.10f   scipy = %.10f" % (lse(v), logsumexp(v)))
print("log P(mode 2)  = v_2 - lse(v) = %.6f" % (v[1] - lse(v)))
v_small = np.array([-800.0, -805.0, -790.0])
with np.errstate(divide="ignore"):
    print("naive log-sum-exp of very negative utilities:",
          np.log(np.exp(v_small).sum()), "  stable:", round(lse(v_small), 6))

# binary logit log-likelihood: log(1 + exp(eta)) via logaddexp
rng = np.random.default_rng(803)
n = 1000
x = rng.normal(0, 1, n)
eta = 40 * x                                     # a very well separated model
y = (rng.uniform(size=n) < expit(eta)).astype(float)
with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
    p = 1 / (1 + np.exp(-eta))
    ll_naive = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
ll_stable = np.sum(y * eta - np.logaddexp(0, eta))
print("binary logit log-likelihood, naive  :", ll_naive)
print("binary logit log-likelihood, stable :", round(ll_stable, 6))
print("log1p(1e-20) =", np.log1p(1e-20), "  log(1 + 1e-20) =", np.log(1 + 1e-20))
