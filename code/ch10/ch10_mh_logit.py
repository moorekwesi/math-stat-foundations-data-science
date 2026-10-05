# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Random-walk Metropolis-Hastings for a Bayesian logistic regression.
# (c) 2026 Stephen E. Moore. MIT Licence.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: one thread
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

rng = np.random.default_rng(803)
# Simulated programme data: enrolment y on intercept, log income, treatment
n = 300
loginc = rng.normal(0, 1, n)
treat = rng.integers(0, 2, n)
X = np.column_stack([np.ones(n), loginc, treat])
beta_true = np.array([-0.5, 0.8, 0.7])
y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta_true))).astype(float)


def log_post(b, tau=10.0):              # N(0, tau^2) prior on each coefficient
    eta = X @ b
    return y @ eta - np.logaddexp(0, eta).sum() - 0.5 * b @ b / tau**2


def ess(x):                             # Geyer initial-positive-sequence ESS
    x = x - x.mean()
    m = len(x)
    f = np.fft.rfft(x, 2 * m)
    acf = np.fft.irfft(f * np.conj(f))[:m] / (x @ x)
    s = 0.0
    for k in range(0, m - 1, 2):        # Gamma_k = rho_2k + rho_2k+1 > 0
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        s += pair
    return m / (2 * s - 1)              # tau = -1 + 2 sum Gamma_k


def rhat(ch):                           # split-R-hat, ch has shape (chains, it)
    h = ch.shape[1] // 2
    s = np.vstack([ch[:, :h], ch[:, h:2 * h]])
    w = s.var(axis=1, ddof=1).mean()
    bvar = h * s.mean(axis=1).var(ddof=1)
    return np.sqrt(((h - 1) / h * w + bvar / h) / w)


# Laplace approximation gives the mode and a proposal covariance
opt = optimize.minimize(lambda b: -log_post(b), np.zeros(3), method="BFGS")
L = np.linalg.cholesky(opt.hess_inv)
scale = 2.38 / np.sqrt(3)
n_iter, burn, chains = 6000, 1000, 4
draws = np.empty((chains, n_iter, 3))
acc = np.zeros(chains)
for c in range(chains):
    b = opt.x + 2 * rng.standard_normal(3)          # over-dispersed start
    lp = log_post(b)
    for t in range(n_iter):
        prop = b + scale * L @ rng.standard_normal(3)
        lp_prop = log_post(prop)
        if np.log(rng.random()) < lp_prop - lp:      # symmetric proposal
            b, lp = prop, lp_prop
            acc[c] += t >= burn
        draws[c, t] = b
post = draws[:, burn:]
print(f"MLE/mode: {np.round(opt.x, 3)}; acceptance rate "
      f"{acc.mean() / (n_iter - burn):.3f}")
print("param     true   post.mean  post.sd   95% cred. int.     ESS   R-hat")
for j, name in enumerate(["const", "loginc", "treat"]):
    flat = post[..., j].ravel()
    lo, hi = np.percentile(flat, [2.5, 97.5])
    e = sum(ess(post[c, :, j]) for c in range(chains))
    print(f"{name:7s} {beta_true[j]:6.2f}  {flat.mean():9.3f} {flat.std():8.3f}"
          f"   [{lo:6.3f}, {hi:6.3f}]  {e:6.0f}  {rhat(post[..., j]):.3f}")
# Policy quantities: P(effect > 0) and average change in enrolment probability
bt = post.reshape(-1, 3)
p1 = 1 / (1 + np.exp(-(bt[:, :2] @ X[:, :2].T + bt[:, [2]])))
p0 = 1 / (1 + np.exp(-(bt[:, :2] @ X[:, :2].T)))
ame = (p1 - p0).mean(axis=1)
print(f"P(treatment coefficient > 0 | data) = {np.mean(bt[:, 2] > 0):.4f}")
print(f"average marginal effect of treatment: {ame.mean():.3f} "
      f"[{np.percentile(ame, 2.5):.3f}, {np.percentile(ame, 97.5):.3f}]")

fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4),
                        gridspec_kw={"width_ratios": [2, 1]})
for c in range(chains):
    axs[0].plot(draws[c, :, 2], lw=0.4)
axs[0].axvline(burn, color="black", ls="--", lw=1)
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("treat coefficient")
axs[0].set_title("trace, 4 chains (dashed: burn-in)", fontsize=10)
xc = post[0, :, 2] - post[0, :, 2].mean()
lags = np.arange(31)
acf = [1.0] + [np.corrcoef(xc[:-k], xc[k:])[0, 1] for k in lags[1:]]
axs[1].bar(lags, acf, color="steelblue")
axs[1].set_xlabel("lag")
axs[1].set_ylabel("autocorrelation")
axs[1].set_title("ACF, chain 1", fontsize=10)
plt.tight_layout()
plt.savefig("ch10_mh_trace.pdf", bbox_inches="tight")
