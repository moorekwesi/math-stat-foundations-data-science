# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Simulation checks of an analytical variance formula and of a derivative.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)

# ---- Part 1: Var(s^2) for iid data with variance sigma2, 4th central moment mu4
#      Var(s^2) = mu4/n - sigma2^2 (n - 3) / (n (n - 1))
def var_s2(n, sigma2, mu4):
    return mu4 / n - sigma2**2 * (n - 3) / (n * (n - 1))

n, reps = 20, 200_000
cases = {"Normal(0,1)": (lambda size: rng.standard_normal(size), 1.0, 3.0),
         "Exponential(1)": (lambda size: rng.exponential(1.0, size), 1.0, 9.0)}
print("Part 1: Var(s^2), n = %d, %d replications" % (n, reps))
print(f"{'population':<16}{'formula':>10}{'normal-only':>13}{'simulated':>11}"
      f"{'MC s.e.':>9}")
for name, (draw, s2, mu4) in cases.items():
    x = draw((reps, n))
    s2_hat = x.var(axis=1, ddof=1)
    sim = s2_hat.var()
    # Monte Carlo s.e. of a variance estimate: sqrt((m4 - v^2) / reps)
    dev2 = (s2_hat - s2_hat.mean())**2
    mcse = np.sqrt(dev2.var() / reps)
    print(f"{name:<16}{var_s2(n, s2, mu4):10.4f}{2 * s2**2 / (n - 1):13.4f}"
          f"{sim:11.4f}{mcse:9.4f}")

# ---- Part 2: gradient of the logistic log-likelihood, analytic vs numerical
m, k = 500, 3
X = np.column_stack([np.ones(m), rng.standard_normal((m, k - 1))])
y = (rng.random(m) < 1 / (1 + np.exp(-X @ np.array([-0.5, 1.0, -2.0])))) * 1.0

def loglik(b):
    eta = X @ b
    return np.sum(y * eta - np.logaddexp(0.0, eta))

def grad(b):
    return X.T @ (y - 1 / (1 + np.exp(-X @ b)))

b0 = np.array([0.2, 0.3, -0.4])
e1 = np.eye(k)
num = np.array([(loglik(b0 + 1e-5 * e) - loglik(b0 - 1e-5 * e)) / 2e-5
                for e in e1])
print("\nPart 2: gradient of logistic log-likelihood at b0")
print("analytic :", np.array2string(grad(b0), precision=6))
print("central  :", np.array2string(num, precision=6))
rel = np.linalg.norm(num - grad(b0)) / np.linalg.norm(grad(b0))
print(f"relative error = {rel:.2e}")

# error of forward / central differences as a function of h (first coordinate)
hs = np.logspace(-12, 0, 49)
g1 = grad(b0)[0]
fwd = [abs((loglik(b0 + h * e1[0]) - loglik(b0)) / h - g1) for h in hs]
cen = [abs((loglik(b0 + h * e1[0]) - loglik(b0 - h * e1[0])) / (2 * h) - g1)
       for h in hs]
print(f"best h: forward {hs[np.argmin(fwd)]:.1e}, central {hs[np.argmin(cen)]:.1e}")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.loglog(hs, fwd, "o-", ms=3, label="forward difference")
ax.loglog(hs, cen, "s-", ms=3, label="central difference")
ax.set_xlabel("step size h")
ax.set_ylabel("absolute error in derivative")
ax.set_title("Truncation error (right) versus rounding error (left)")
ax.legend()
ax.grid(True, which="major", alpha=0.3)
plt.savefig("ch13_fd_error.pdf", bbox_inches="tight")
