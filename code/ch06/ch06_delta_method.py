# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Delta-method standard errors for a ratio of means and an arc elasticity.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
reps, n = 5000, 400


def survey(size):
    """Simulated households: total expenditure X and food expenditure Y."""
    x = np.exp(rng.normal(6.5, 0.6, size))
    y = 0.15 * x + 3 * np.sqrt(x) * np.exp(rng.normal(0, 0.3, size))
    return x, y


# 1. Food share R = E[Y]/E[X]; gradient of g(a, b) = b/a is (-b/a^2, 1/a)
R_hat, se_hat = np.empty(reps), np.empty(reps)
for r in range(reps):
    x, y = survey(n)
    mx, my = x.mean(), y.mean()
    S = np.cov(x, y)
    grad = np.array([-my / mx**2, 1 / mx])
    R_hat[r] = my / mx
    se_hat[r] = np.sqrt(grad @ S @ grad / n)
xb, yb = survey(2_000_000)
R0 = yb.mean() / xb.mean()
cover = np.mean(np.abs(R_hat - R0) <= 1.96 * se_hat)
print("Ratio of means (aggregate food share), n = 400")
print(f"  true R = {R0:.4f}, mean of R_hat = {R_hat.mean():.4f}")
print(f"  Monte Carlo sd of R_hat = {R_hat.std():.5f}")
print(f"  mean delta-method se    = {se_hat.mean():.5f}")
print(f"  coverage of R_hat +- 1.96 se = {cover:.3f}")

# 2. Arc price elasticity from two independent survey rounds
p0, p1 = 10.0, 12.0                          # prices (known)
q0_mu, q1_mu, sd_q = 50.0, 46.0, 15.0        # mean quantities, sd
eta0 = np.log(q1_mu / q0_mu) / np.log(p1 / p0)
q0 = rng.normal(q0_mu, sd_q, size=(reps, n)).mean(axis=1)
q1 = rng.normal(q1_mu, sd_q, size=(reps, n)).mean(axis=1)
eta = np.log(q1 / q0) / np.log(p1 / p0)
se_delta = np.sqrt(sd_q**2 / (n * q1_mu**2) + sd_q**2 / (n * q0_mu**2))
se_delta /= np.log(p1 / p0)
print("\nArc price elasticity  eta = log(q1/q0)/log(p1/p0)")
print(f"  true eta = {eta0:.4f}, mean of eta_hat = {eta.mean():.4f}")
print(f"  Monte Carlo sd = {eta.std():.4f}, delta-method se = {se_delta:.4f}")
print(f"  95% interval: {eta0:.3f} +- {1.96 * se_delta:.3f}")
