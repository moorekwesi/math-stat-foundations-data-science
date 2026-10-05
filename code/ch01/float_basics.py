# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# IEEE 754 double precision: representation, spacing, overflow and underflow.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3)
print("0.1 + 0.2 =", repr(0.1 + 0.2), "  0.1 stored as", f"{0.1:.20f}")
print("0.1 in hexadecimal:", (0.1).hex())
for t in (np.float64, np.float32):
    fi = np.finfo(t)
    print(f"{t.__name__:8s} eps = {fi.eps:.3e}  max = {fi.max:.3e}"
          f"  tiny = {fi.tiny:.3e}  digits = {fi.precision}")
print("unit roundoff u = eps/2 = %.3e" % (np.finfo(float).eps / 2))

# spacing between neighbouring doubles grows with magnitude
for x in [1.0, 1e3, 1e8, 1e16]:
    print(f"spacing(ulp) at {x:8.0e}: {np.spacing(x):.3e}")
print("1e16 + 1 - 1e16 =", 1e16 + 1 - 1e16)
print("2**53 + 1 == 2**53 as floats ?", float(2**53) + 1 == float(2**53))
print("1 + eps/2 == 1 ?", 1.0 + np.finfo(float).eps / 2 == 1.0)

# overflow, underflow and special values
with np.errstate(over="ignore", invalid="ignore"):
    big = np.exp(np.array([709.0, 710.0]))
    print("exp(709), exp(710) =", big)
    print("inf - inf =", big[1] - big[1], "  1/inf =", 1 / big[1])
print("5e-324 (smallest subnormal) =", 5e-324, "  5e-324 / 2 =", 5e-324 / 2)
print("nan == nan ?", np.nan == np.nan, "  np.isnan(nan):", np.isnan(np.nan))

# integers do not overflow gracefully either
a = np.array([2**62], dtype=np.int64)
print("int64: 2**62 * 2 =", (a * 2)[0], "  (wrap-around, no warning)")
print("40000.7 cast to int16 ->", np.array([40000.7]).astype(np.int16)[0],
      "  (cf. Ariane 5)")

# comparing floats
print("np.isclose(0.1 + 0.2, 0.3):", np.isclose(0.1 + 0.2, 0.3))
print("np.isclose(1e-10, 2e-10)  :", np.isclose(1e-10, 2e-10),
      " <- default atol=1e-8 dominates!")
print("np.isclose(1e-10, 2e-10, atol=0):", np.isclose(1e-10, 2e-10, atol=0))
