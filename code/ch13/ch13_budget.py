# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Budget allocation under uncertain concave returns: Monte Carlo + SLSQP.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(803)
names = ["cash", "school", "health", "training"]
m_a = np.array([1.6, 2.2, 3.0, 2.6])        # mean initial marginal benefit
s_a = np.array([0.10, 0.25, 0.50, 0.70])    # log-sd: evidence quality differs
m_c = np.array([60.0, 30.0, 15.0, 25.0])    # saturation scales
S, B = 4000, 100.0                          # Monte Carlo draws, budget

# Parameter draws (common random numbers, reused for every allocation)
z = rng.standard_normal((S, 4))
a = m_a * np.exp(s_a * z - 0.5 * s_a**2)                  # E[a_j] = m_a[j]
c = m_c * np.exp(0.3 * rng.standard_normal((S, 4)) - 0.045)


def benefit_draws(x):
    """Total benefit in each draw: sum_j a_j c_j log(1 + x_j / c_j)."""
    return np.sum(a * c * np.log1p(x / c), axis=1)


def objective(x, kappa):
    """Negative of mean - kappa * sd, with its gradient."""
    tot = benefit_draws(x)
    marg = a / (1 + x / c)                                # d benefit / d x_j
    mean, sd = tot.mean(), tot.std()
    g_mean = marg.mean(0)
    g_sd = ((tot - mean)[:, None] * (marg - g_mean)).mean(0) / sd
    return -(mean - kappa * sd), -(g_mean - kappa * g_sd)


def solve(kappa, budget=B, params=None):
    cons = {"type": "eq", "fun": lambda x: x.sum() - budget,
            "jac": lambda x: np.ones(4)}
    res = minimize(objective, np.full(4, budget / 4), args=(kappa,), jac=True,
                   method="SLSQP", bounds=[(0, budget)] * 4, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x, -res.fun


def summary(x):
    tot = benefit_draws(x)
    return tot.mean(), tot.std(), np.quantile(tot, 0.05)


# Plug-in solution: ignore uncertainty, use mean parameters (closed form check)
def plug_in(budget=B):
    obj = lambda x: (-np.sum(m_a * m_c * np.log1p(x / m_c)),
                     -m_a / (1 + x / m_c))
    cons = {"type": "eq", "fun": lambda x: x.sum() - budget}
    return minimize(obj, np.full(4, budget / 4), jac=True, method="SLSQP",
                    bounds=[(0, budget)] * 4, constraints=cons).x


print(f"budget B = {B:.0f}, S = {S} Monte Carlo draws of (a, c)")
print(f"{'allocation':<22}" + "".join(f"{nm:>9}" for nm in names)
      + f"{'mean':>8}{'sd':>7}{'P5':>7}")
rows = [("plug-in (means)", plug_in()), ("max E[benefit]", solve(0.0)[0]),
        ("mean - 1.0 sd", solve(1.0)[0]), ("mean - 2.0 sd", solve(2.0)[0])]
for lab, x in rows:
    mn, sd, p5 = summary(x)
    print(f"{lab:<22}" + "".join(f"{v:9.1f}" for v in x)
          + f"{mn:8.1f}{sd:7.1f}{p5:7.1f}")

# Shadow price of the budget: KKT multiplier vs finite difference of V(B)
x0, V0 = solve(0.0)
marg = (a / (1 + x0 / c)).mean(0)
print("\nexpected marginal benefit per unit:", np.array2string(marg, precision=3))
print(f"KKT multiplier (funded programmes): {marg[x0 > 1e-6].mean():.4f}")
dB = 1.0
Vp, Vm = solve(0.0, B + dB)[1], solve(0.0, B - dB)[1]
print(f"finite difference dV/dB          : {(Vp - Vm) / (2 * dB):.4f}")

# Monte Carlo error of the optimal value: split draws into 10 batches
batches = benefit_draws(x0).reshape(10, -1).mean(1)
print(f"MC s.e. of expected benefit at optimum: {batches.std(ddof=1) / 10**0.5:.3f}")

# Sensitivity: optimal (risk-neutral) allocation as the budget varies
budgets = np.linspace(10, 250, 25)
alloc = np.array([solve(0.0, bb)[0] for bb in budgets])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.stackplot(budgets, alloc.T / budgets, labels=names, alpha=0.85)
ax.set_xlabel("total budget B")
ax.set_ylabel("share of budget")
ax.set_xlim(budgets[0], budgets[-1])
ax.set_ylim(0, 1)
ax.set_title("Risk-neutral optimal allocation shares")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
plt.savefig("ch13_budget_shares.pdf", bbox_inches="tight")
