# DEP 803 -- Chapter 8: Optimisation
# Gauss-Newton for a logistic adoption curve (simulated mobile-money uptake)
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

rng = np.random.default_rng(803)
t = np.arange(48.0)                                  # months since launch
K, r, t0 = 0.62, 0.18, 22.0                          # true parameters
share = K / (1 + np.exp(-r * (t - t0))) + rng.normal(0, 0.015, t.size)


def model(th):
    return th[0] / (1 + np.exp(-th[1] * (t - th[2])))


def resid(th):
    return share - model(th)


def jac(th):                                         # Jacobian of residuals
    e = np.exp(-th[1] * (t - th[2]))
    d = 1 + e
    return -np.column_stack([1 / d, th[0] * (t - th[2]) * e / d**2,
                             -th[0] * th[1] * e / d**2])


start = np.array([0.5, 0.1, 15.0])                   # starting guess
th = start.copy()
print(f"{'it':>3}{'K':>9}{'r':>9}{'t0':>9}{'SSR':>12}")
for k in range(15):
    rk, J = resid(th), jac(th)
    print(f"{k:>3}{th[0]:>9.4f}{th[1]:>9.4f}{th[2]:>9.4f}{rk @ rk:>12.4e}")
    step = np.linalg.lstsq(J, -rk, rcond=None)[0]   # solve J d = -r (LS)
    th = th + step
    if np.linalg.norm(step) < 1e-9:
        break
lm = least_squares(resid, start, jac=jac, method="lm")
print("scipy least_squares (LM):", np.round(lm.x, 4), " nfev =", lm.nfev)

plt.figure(figsize=(6, 3.8))
plt.plot(t, share, "o", ms=4, label="simulated adoption share")
plt.plot(t, model(th), lw=2, label="Gauss-Newton fit")
plt.plot(t, model(start), "--", label="starting curve")
plt.gca().set(xlabel="months since launch", ylabel="share of adults using service",
              title="Nonlinear least squares: logistic diffusion curve")
plt.legend(frameon=False)
plt.savefig("ch08_gauss_newton.pdf", bbox_inches="tight")
