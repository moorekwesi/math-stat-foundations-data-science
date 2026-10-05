# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# A minimal Hamiltonian Monte Carlo sampler versus random-walk Metropolis.
# (c) 2026 Stephen E. Moore. MIT Licence.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: one thread
import numpy as np

rng = np.random.default_rng(803)
d, rho = 10, 0.9                         # correlated 10-dimensional Gaussian
S = rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))   # AR(1) cov
P = np.linalg.inv(S)
logp = lambda x: -0.5 * x @ P @ x
grad = lambda x: -P @ x


def ess(x):
    x = x - x.mean()
    m = len(x)
    f = np.fft.rfft(x, 2 * m)
    acf = np.fft.irfft(f * np.conj(f))[:m] / (x @ x)
    s = 0.0
    for k in range(0, m - 1, 2):
        if acf[k] + acf[k + 1] < 0:
            break
        s += acf[k] + acf[k + 1]
    return m / (2 * s - 1)


def hmc(n, eps=0.15, L=20):
    x, out, acc = np.zeros(d), np.empty((n, d)), 0
    for t in range(n):
        p = rng.standard_normal(d)
        xn, pn = x.copy(), p + 0.5 * eps * grad(x)      # leapfrog integrator
        for i in range(L):
            xn = xn + eps * pn
            if i < L - 1:
                pn = pn + eps * grad(xn)
        pn = pn + 0.5 * eps * grad(xn)
        dH = logp(xn) - 0.5 * pn @ pn - logp(x) + 0.5 * p @ p
        if np.log(rng.random()) < dH:
            x, acc = xn, acc + 1
        out[t] = x
    return out, acc / n


def rwm(n, scale):
    x, out, acc = np.zeros(d), np.empty((n, d)), 0
    lp = logp(x)
    for t in range(n):
        prop = x + scale * rng.standard_normal(d)
        lq = logp(prop)
        if np.log(rng.random()) < lq - lp:
            x, lp, acc = prop, lq, acc + 1
        out[t] = x
    return out, acc / n


n_h, L = 2000, 20
xh, ah = hmc(n_h, L=L)
xr, ar = rwm(n_h * L, 2.38 / np.sqrt(d) * 0.45)    # same gradient budget
for name, x, a in [("HMC", xh, ah), ("RW-Metropolis", xr, ar)]:
    e = np.array([ess(x[:, j]) for j in range(d)])
    print(f"{name:14s} iterations {len(x):6d}  acceptance {a:.3f}  "
          f"min/median ESS {e.min():6.0f} / {np.median(e):6.0f}")
    print(f"{'':14s} var(x1) {x[:, 0].var():.3f} (target 1),  "
          f"ESS per 1000 density evaluations {1000 * e.min() / (n_h * L):.1f}")
