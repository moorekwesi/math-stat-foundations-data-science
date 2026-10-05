# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Numerical checks of the ten worked review problems (Chapters 1-12).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
rows = []

# R1 cancellation: 1 - cos(x) versus 2 sin^2(x/2) at x = 1e-8
x = 1e-8
rows.append(("R1 1-cos(1e-8)", 0.5 * x**2, 1 - np.cos(x), 2 * np.sin(x / 2)**2))

# R2 hat matrix: trace = k, cond(X'X) = cond(X)^2
X = np.column_stack([np.ones(50), rng.standard_normal((50, 3))])
H = X @ np.linalg.solve(X.T @ X, X.T)
rows.append(("R2 trace(H)", 4, np.trace(H), np.allclose(H @ H, H)))
rows.append(("R2 cond(X'X)/cond^2", 1.0,
             np.linalg.cond(X.T @ X) / np.linalg.cond(X)**2, np.nan))

# R3 equicorrelation p = 10, rho = 0.3: largest eigenvalue 1 + 9 rho
p, rho = 10, 0.3
C = (1 - rho) * np.eye(p) + rho * np.ones((p, p))
lam = np.linalg.eigvalsh(C)
rows.append(("R3 PC1 share", (1 + (p - 1) * rho) / p, lam[-1] / p, lam[0]))

# R4 inverse transform -log(U)/2 ~ Exp(rate 2); MLE of the rate = 1/mean
u = rng.random(100_000)
e = -np.log(u) / 2
ks = stats.kstest(e, "expon", args=(0, 0.5)).pvalue
rows.append(("R4 rate MLE", 2.0, 1 / e.mean(), ks))

# R5 Hoeffding / Chebyshev sample sizes for eps = 0.05, delta = 0.05
rows.append(("R5 n Hoeffding", 738, np.ceil(np.log(2 / 0.05) / (2 * 0.05**2)),
             np.nan))
rows.append(("R5 n Chebyshev", 2000, np.ceil(0.25 / (0.05**2 * 0.05)), np.nan))

# R6 Uniform(0, theta): MSE of 2*mean and of the max, n = 10, theta = 1
n, R = 10, 200_000
U = rng.random((R, n))
rows.append(("R6 MSE(2 xbar)", 1 / (3 * n), np.mean((2 * U.mean(1) - 1)**2),
             np.nan))
rows.append(("R6 MSE(max)", 2 / ((n + 1) * (n + 2)), np.mean((U.max(1) - 1)**2),
             np.nan))

# R7 Newton on f(x) = exp(x) - 2x from x0 = 0
xk = 0.0
for _ in range(5):
    xk -= (np.exp(xk) - 2) / np.exp(xk)
rows.append(("R7 Newton argmin", np.log(2), xk, abs(xk - np.log(2))))

# R8 max sum a_j log x_j s.t. sum x_j = B: x_j = a_j B / sum(a)
a, B = np.array([1.0, 2.0, 5.0]), 80.0
xs = a * B / a.sum()
rows.append(("R8 x_3 and lambda", 50.0, xs[2], a.sum() / B))

# R9 plain vs antithetic MC for int_0^1 e^x dx; bootstrap inclusion prob.
m = 100_000
u = rng.random(m)
plain = np.exp(u)
anti = 0.5 * (np.exp(u[: m // 2]) + np.exp(1 - u[: m // 2]))
rows.append(("R9 Var(e^U)", (np.e**2 - 1) / 2 - (np.e - 1)**2, plain.var(),
             anti.var()))
nn = 500
inc = np.mean([np.unique(rng.integers(0, nn, nn)).size / nn for _ in range(2000)])
rows.append(("R9 P(in boot sample)", 1 - (1 - 1 / nn)**nn, inc, 1 - np.exp(-1)))

# R10 ridge shrinkage factors d^2/(d^2+lambda), singular values 3 and 1, lam = 1
d = np.array([3.0, 1.0])
Uo, _, Vo = np.linalg.svd(rng.standard_normal((20, 2)), full_matrices=False)
Xr = Uo * d @ Vo
Hr = Xr @ np.linalg.solve(Xr.T @ Xr + np.eye(2), Xr.T)
rows.append(("R10 ridge df", 0.9 + 0.5, np.trace(Hr), np.nan))

print(f"{'check':<22}{'analytical':>14}{'numerical':>14}{'extra':>14}")
for name, an, nu, ex in rows:
    if isinstance(ex, (bool, np.bool_)):
        exs = f"{str(ex):>14}"
    else:
        exs = f"{'--':>14}" if np.isnan(ex) else f"{ex:14.4g}"
    print(f"{name:<22}{an:14.6g}{nu:14.6g}{exs}")
