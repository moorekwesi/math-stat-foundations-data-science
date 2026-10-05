# DEP 803 -- Appendix A: Python Toolkit
# SciPy and scikit-learn essentials: distributions, optimisation, integration.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats, optimize, integrate
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(803)

# Distributions: pdf, cdf, quantiles and random variates
Z = stats.norm(loc=0, scale=1)
print(f"P(Z <= 1.96) = {Z.cdf(1.96):.4f},  97.5% quantile = {Z.ppf(0.975):.4f}")
t5 = stats.t(df=5)
print(f"97.5% quantile of t(5) = {t5.ppf(0.975):.4f}")
sample = stats.gamma(a=2.0, scale=1.5).rvs(size=2000, random_state=rng)
a_hat, loc_hat, scale_hat = stats.gamma.fit(sample, floc=0)
print(f"gamma MLE: shape = {a_hat:.3f}, scale = {scale_hat:.3f}  (true 2, 1.5)")

# Optimisation: minimise the Rosenbrock function with BFGS
res = optimize.minimize(optimize.rosen, x0=[-1.2, 1.0], method="BFGS",
                        jac=optimize.rosen_der)
print("BFGS minimiser:", np.round(res.x, 6), " iterations:", res.nit)

# Constrained optimisation: maximise sqrt(x1) + sqrt(x2) s.t. x1 + 2 x2 = 10
cons = {"type": "eq", "fun": lambda v: v[0] + 2 * v[1] - 10}
res = optimize.minimize(lambda v: -(np.sqrt(v[0]) + np.sqrt(v[1])),
                        x0=[1.0, 1.0], method="SLSQP", constraints=[cons],
                        bounds=[(1e-9, None), (1e-9, None)],
                        options={"ftol": 1e-12})
print("SLSQP solution:", np.round(res.x, 4), " (exact 6.6667, 1.6667)")

# Integration: E[exp(Z)] for Z ~ N(0,1) equals exp(1/2)
integrand = lambda z: np.exp(z - z**2 / 2) / np.sqrt(2 * np.pi)   # exp(z) phi(z)
val, err = integrate.quad(integrand, -np.inf, np.inf)
print(f"quad: {val:.10f}   exact: {np.exp(0.5):.10f}")

# scikit-learn: the fit/predict interface
X = rng.normal(size=(500, 3))
y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.5, 500)
model = LinearRegression().fit(X, y)
print("sklearn coefficients:", np.round(model.coef_, 3),
      " R^2 =", round(model.score(X, y), 3))
