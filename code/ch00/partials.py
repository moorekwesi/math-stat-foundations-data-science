# DEP 803 -- Chapter 0: Mathematical Refresher
# Partial derivatives and the gradient by finite differences; a two-good firm.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np


def f(x, y):
    return x**2 * y + 3 * x * y**2


def grad_fd(func, x, y, h=1e-6):
    """Central finite-difference approximation of the gradient."""
    fx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    fy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return np.array([fx, fy])


print("f(x, y) = x^2 y + 3 x y^2 at (1, 2)")
print("exact gradient (2xy + 3y^2, x^2 + 6xy) =", (2 * 1 * 2 + 3 * 4, 1 + 6 * 2))
print("finite differences                    =", grad_fd(f, 1.0, 2.0).round(6))


# two-product firm: prices 10 and 12, cost C = x^2 + xy + y^2
def profit(x, y):
    return 10 * x + 12 * y - (x**2 + x * y + y**2)


# FOCs: 10 - 2x - y = 0 and 12 - x - 2y = 0  ->  solve the 2x2 system
A = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([10.0, 12.0])
x_star, y_star = np.linalg.solve(A, b)
print(f"\nprofit-maximising outputs: x* = {x_star:.4f}, y* = {y_star:.4f}")
print("gradient at the optimum   :", grad_fd(profit, x_star, y_star).round(6) + 0.0)
print("maximum profit            :", round(profit(x_star, y_star), 4))
H = np.array([[-2.0, -1.0], [-1.0, -2.0]])   # second derivatives (Hessian)
print("second-order check: f_xx =", H[0, 0], " det(H) =", round(np.linalg.det(H), 4))
