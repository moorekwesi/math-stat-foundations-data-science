# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Composite trapezoidal and Simpson rules: observed convergence rates.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def trapezoid(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (y.sum() - 0.5 * (y[0] + y[-1]))


def simpson(f, a, b, n):          # n must be even
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


cases = [("exp(x) on [0,1]", np.exp, np.e - 1),
         ("sqrt(x) on [0,1]", np.sqrt, 2.0 / 3.0)]
ns = 2 ** np.arange(1, 11)
fig, ax = plt.subplots(figsize=(6, 3.8))
for (label, f, exact), ls in zip(cases, ["-", "--"]):
    et = np.array([abs(trapezoid(f, 0, 1, n) - exact) for n in ns])
    es = np.array([abs(simpson(f, 0, 1, n) - exact) for n in ns])
    print(f"{label}")
    print("     n   trap error  ratio   simp error  ratio")
    for i, n in enumerate(ns[:6]):
        rt = et[i - 1] / et[i] if i else np.nan
        rs = es[i - 1] / es[i] if i else np.nan
        print(f"{n:6d}   {et[i]:9.2e}  {rt:5.2f}   {es[i]:9.2e}  {rs:5.2f}")
    slope_t = np.polyfit(np.log(ns), np.log(et), 1)[0]
    slope_s = np.polyfit(np.log(ns[:6]), np.log(es[:6]), 1)[0]
    print(f"   fitted log-log slopes: trapezoid {slope_t:.2f}, "
          f"Simpson {slope_s:.2f}")
    ax.loglog(ns, et, "o" + ls, color="navy", ms=4, label=f"trapezoid, {label}")
    ax.loglog(ns, np.maximum(es, 1e-17), "s" + ls, color="darkorange", ms=4,
              label=f"Simpson, {label}")
ax.set_xlabel("number of subintervals n")
ax.set_ylabel("absolute error")
ax.legend(fontsize=8)
plt.savefig("ch10_newton_cotes.pdf", bbox_inches="tight")
