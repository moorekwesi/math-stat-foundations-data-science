# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Python loops versus vectorised NumPy, and the O(n^3) cost of matrix products.
# (c) 2026 Stephen E. Moore. MIT Licence.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # one thread: clean scaling
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
rng = np.random.default_rng(803)


def best_time(f, repeat=3):
    """Best wall-clock time of several runs (reduces timing noise)."""
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        out.append(time.perf_counter() - t0)
    return min(out)


x = rng.standard_normal(1_000_000)
t_loop = best_time(lambda: sum(xi * xi for xi in x), repeat=1)
t_vec = best_time(lambda: x @ x)
print(f"sum of squares, n = 1e6: loop {t_loop:.3f}s, vectorised {t_vec:.5f}s,"
      f" speed-up ~{t_loop / t_vec:,.0f}x")

sizes = np.array([200, 400, 800, 1200, 1600])
t_mm, t_solve = [], []
for n in sizes:
    M = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    np.linalg.solve(M, b)                     # warm-up call
    t_mm.append(best_time(lambda: M @ M, repeat=5))
    t_solve.append(best_time(lambda: np.linalg.solve(M, b), repeat=5))
t_mm, t_solve = np.array(t_mm), np.array(t_solve)
print(f"{'n':>6s} {'matmul (s)':>11s} {'GFLOP/s':>8s} {'solve (s)':>10s}")
for n, a, s in zip(sizes, t_mm, t_solve):
    print(f"{n:6d} {a:11.4f} {2 * float(n)**3 / a / 1e9:8.1f} {s:10.4f}")
slope = np.polyfit(np.log(sizes[1:]), np.log(t_mm[1:]), 1)[0]
print(f"fitted log-log slope for matmul time: {slope:.2f} (theory: 3)")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.loglog(sizes, t_mm, "o-", label="matrix product A @ B  (2n^3 flops)")
ax.loglog(sizes, t_solve, "s-", label="solve Ax = b  (2n^3/3 flops)")
ref = t_mm[-1] * (sizes / sizes[-1]) ** 3
ax.loglog(sizes, ref, "k--", lw=1, label="slope 3 reference")
ax.set_xlabel("matrix dimension n")
ax.set_ylabel("seconds (best of 5)")
ax.set_title("Dense linear algebra cost grows like n^3")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch01_timing.pdf", bbox_inches="tight")
