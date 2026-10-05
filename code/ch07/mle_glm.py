# DEP 803 -- Chapter 7: Estimation Principles
# Numerical MLE: logit and probit take-up models; Poisson demand with exposure.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import optimize, stats, special
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(803)
n = 2000
# Simulated programme take-up survey (NOT real data)
income = rng.normal(0, 1, n)                    # standardised log income
dist = rng.uniform(0, 5, n)                     # km to enrolment office
female = rng.binomial(1, 0.5, n)                # female-headed household
X = np.column_stack([np.ones(n), income, dist, female])
b_true = np.array([0.5, -0.6, -0.4, 0.5])
y = rng.binomial(1, special.expit(X @ b_true))


def logit_nll(b):
    eta = X @ b
    return np.sum(np.logaddexp(0, eta) - y * eta)   # stable log(1 + e^eta)


def logit_grad(b):
    return X.T @ (special.expit(X @ b) - y)


def probit_nll(b):
    eta = X @ b
    return -np.sum(y * stats.norm.logcdf(eta)
                   + (1 - y) * stats.norm.logcdf(-eta))


b0 = np.zeros(4)
r_logit = optimize.minimize(logit_nll, b0, jac=logit_grad, method="BFGS")
r_probit = optimize.minimize(probit_nll, b0, method="BFGS")
sk = LogisticRegression(penalty="none", max_iter=1000).fit(X[:, 1:], y)
sk_b = np.r_[sk.intercept_, sk.coef_.ravel()]
names = ["const", "income", "distance", "female"]
print(f"Take-up model (simulated), n = {n}")
print(f"{'':>9} {'true':>7} {'logit':>8} {'sklearn':>8} {'probit':>8}"
      f" {'1.6*probit':>10}")
for j, nm in enumerate(names):
    print(f"{nm:>9} {b_true[j]:7.2f} {r_logit.x[j]:8.4f} {sk_b[j]:8.4f}"
          f" {r_probit.x[j]:8.4f} {1.6 * r_probit.x[j]:10.4f}")
print(f"log-lik: logit = {-r_logit.fun:.2f}, probit = {-r_probit.fun:.2f}")
# Average marginal effect of distance on the take-up probability
p = special.expit(X @ r_logit.x)
ame_l = np.mean(p * (1 - p)) * r_logit.x[2]
ame_p = np.mean(stats.norm.pdf(X @ r_probit.x)) * r_probit.x[2]
print(f"AME of +1 km distance: logit {ame_l:.4f}, probit {ame_p:.4f}")

# Poisson regression with exposure: clinic visits in m districts
m = 120
pop = rng.integers(5, 80, m) * 1000              # district population
urban = rng.binomial(1, 0.4, m)
fee = rng.uniform(0, 2, m)                       # user fee (simulated units)
Xd = np.column_stack([np.ones(m), urban, fee])
g_true = np.array([-1.0, 0.3, -0.35])            # log visits per person-year
visits = rng.poisson(pop * np.exp(Xd @ g_true))
off = np.log(pop)


def pois_nll(g):
    eta = off + Xd @ g
    return np.sum(np.exp(eta) - visits * eta)


def pois_grad(g):
    return Xd.T @ (np.exp(off + Xd @ g) - visits)


r_p = optimize.minimize(pois_nll, np.zeros(3), jac=pois_grad, method="BFGS")
print(f"\nPoisson demand with log(population) offset, m = {m} districts")
for nm, t, e in zip(["const", "urban", "fee"], g_true, r_p.x):
    print(f"{nm:>9}: true {t:6.2f}, MLE {e:8.4f}, rate ratio {np.exp(e):.4f}")
