# DEP 803 -- Chapter 8: Optimisation
# M-estimators (OLS, LAD, Huber) on a simulated earnings survey with outliers
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(803)
n = 400
educ = rng.uniform(0, 16, n)                       # years of schooling
logw = 1.0 + 0.08 * educ + rng.normal(0, 0.3, n)   # simulated log earnings
bad = rng.choice(n, 20, replace=False)             # 5% gross recording errors
logw[bad] += rng.uniform(2.0, 4.0, 20)
X = np.column_stack([np.ones(n), educ])


def huber(r, c=1.345):
    a = np.abs(r)
    return np.where(a <= c, 0.5 * r**2, c * a - 0.5 * c**2)


losses = {
    "OLS (squared)": lambda r: 0.5 * r**2,
    "LAD (absolute)": lambda r: np.abs(r),
    "Huber (c=1.345)": lambda r: huber(r / 0.3),   # residual scaled by sigma
}
b0 = np.linalg.lstsq(X, logw, rcond=None)[0]       # starting value
print(f"{'estimator':<18}{'intercept':>10}{'slope':>9}{'iters':>7}{'risk':>10}")
print(f"{'truth':<18}{1.0:>10.4f}{0.08:>9.4f}")
for name, rho in losses.items():
    risk = lambda b, rho=rho: np.mean(rho(logw - X @ b))
    res = minimize(risk, b0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    b = res.x
    print(f"{name:<18}{b[0]:>10.4f}{b[1]:>9.4f}{res.nit:>7d}{res.fun:>10.5f}")

# figure: the three loss functions rho(r)
r = np.linspace(-3, 3, 400)
plt.figure(figsize=(6, 3.8))
plt.plot(r, 0.5 * r**2, label=r"squared $\frac{1}{2}r^2$")
plt.plot(r, np.abs(r), label=r"absolute $|r|$")
plt.plot(r, huber(r), label="Huber, c = 1.345")
plt.ylim(0, 4)
plt.gca().set(xlabel="residual r", ylabel=r"loss $\rho(r)$",
              title="Loss functions behind three M-estimators")
plt.legend(frameon=False)
plt.savefig("ch08_losses.pdf", bbox_inches="tight")
