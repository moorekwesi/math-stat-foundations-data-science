# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Monte Carlo bias-variance decomposition for polynomial regression.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre

rng = np.random.default_rng(803)
f = lambda x: np.sin(1.5 * np.pi * x) + 0.5 * x**2    # true regression function
sigma, n, R = 0.35, 50, 2000
x0 = np.linspace(-0.95, 0.95, 60)                     # test inputs
degrees = np.arange(0, 11)


def design(x, deg):                                  # well-conditioned basis
    return legendre.legvander(x, deg)


xs = rng.uniform(-1, 1, (R, n))                     # R training samples
ys = f(xs) + sigma * rng.normal(size=(R, n))
y0 = f(x0) + sigma * rng.normal(size=(R, x0.size))  # fresh test responses
bias2, var, mse = [], [], []
for deg in degrees:
    Xs = design(xs, deg)                             # (R, n, deg + 1)
    coef = np.linalg.pinv(Xs) @ ys[:, :, None]       # batched least squares
    preds = (design(x0, deg) @ coef)[:, :, 0]        # (R, 60)
    bias2.append(np.mean((preds.mean(0) - f(x0)) ** 2))
    var.append(np.mean(preds.var(0)))
    mse.append(np.mean((y0 - preds) ** 2))
bias2, var, mse = map(np.array, (bias2, var, mse))

print(f"{'deg':>3}{'bias^2':>9}{'variance':>10}{'noise':>8}"
      f"{'sum':>9}{'MC test MSE':>13}")
for k in range(len(degrees)):
    print(f"{degrees[k]:>3}{bias2[k]:>9.4f}{var[k]:>10.4f}{sigma**2:>8.4f}"
          f"{bias2[k] + var[k] + sigma**2:>9.4f}{mse[k]:>13.4f}")
print("degree minimising test MSE:", degrees[np.argmin(mse)])

plt.figure(figsize=(6, 3.8))
plt.plot(degrees, bias2, "o-", label="squared bias")
plt.plot(degrees, var, "s-", label="variance")
plt.plot(degrees, mse, "k^-", label="expected test MSE")
plt.axhline(sigma**2, color="gray", ls="--", label=r"noise $\sigma^2$")
plt.yscale("log")
plt.xlabel("polynomial degree (model complexity)")
plt.ylabel("error (log scale)")
plt.title(f"Bias-variance decomposition, n = {n}, {R} replications")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.savefig("ch12_bias_variance.pdf", bbox_inches="tight")
