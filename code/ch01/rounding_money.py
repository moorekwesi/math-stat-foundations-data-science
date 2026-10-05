# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Rounding rules for money and published statistics: binary floats vs decimal.
# (c) 2026 Stephen E. Moore. MIT Licence.
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import numpy as np

print("round(0.5), round(1.5), round(2.5) =", round(0.5), round(1.5), round(2.5),
      "  (round half to even)")
print("round(2.675, 2) =", round(2.675, 2), "  because 2.675 is stored as",
      f"{2.675:.20f}")
d = Decimal("2.665")
print("Decimal('2.665') half-up   ->", d.quantize(Decimal("0.01"), ROUND_HALF_UP))
print("Decimal('2.665') half-even ->", d.quantize(Decimal("0.01"), ROUND_HALF_EVEN))
print("Decimal(2.675) (from float) =", Decimal(2.675))

# VAT of 15 percent on 10,000 simulated invoice lines (amounts in pesewas)
rng = np.random.default_rng(803)
pesewas = rng.integers(100, 50_000, 10_000)               # exact integers
rate = Decimal("0.15")
cent = Decimal("1")
line_tax = sum((Decimal(int(a)) * rate).quantize(cent, ROUND_HALF_UP)
               for a in pesewas)
total_tax = (Decimal(int(pesewas.sum())) * rate).quantize(cent, ROUND_HALF_UP)
print(f"\nVAT rounded per line   : {line_tax} pesewas")
print(f"VAT rounded on total   : {total_tax} pesewas")
print(f"difference             : {line_tax - total_tax} pesewas")

# published growth rates: rounding components versus rounding the total
w = np.array([0.30, 0.25, 0.25, 0.20])               # sector weights
g = np.array([2.346, 4.151, 3.849, 1.249])           # sector growth (%)
print("\nsector contributions, rounded to 1 dp:", np.round(w * g, 1),
      " sum =", round(float(np.round(w * g, 1).sum()), 1))
print("aggregate growth rounded to 1 dp     :", round(float(w @ g), 1))
