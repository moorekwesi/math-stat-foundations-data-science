# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Forward mode (dual numbers) and reverse mode (tape) automatic differentiation.
# (c) 2026 Stephen E. Moore. MIT Licence.
import math


class Dual:
    """Forward mode: carry (value, directional derivative)."""
    def __init__(self, v, d=0.0):
        self.v, self.d = v, d

    def __add__(self, o):
        return Dual(self.v + o.v, self.d + o.d)

    def __sub__(self, o):
        return Dual(self.v - o.v, self.d - o.d)

    def __mul__(self, o):
        return Dual(self.v * o.v, self.d * o.v + self.v * o.d)


def dlog(a):
    return Dual(math.log(a.v), a.d / a.v)


def dsin(a):
    return Dual(math.sin(a.v), math.cos(a.v) * a.d)


class Var:
    """Reverse mode: record parents and local partials, then sweep back."""
    def __init__(self, v, parents=()):
        self.v, self.parents, self.grad = v, parents, 0.0

    def __add__(self, o):
        return Var(self.v + o.v, ((self, 1.0), (o, 1.0)))

    def __sub__(self, o):
        return Var(self.v - o.v, ((self, 1.0), (o, -1.0)))

    def __mul__(self, o):
        return Var(self.v * o.v, ((self, o.v), (o, self.v)))

    def backward(self):
        order, seen = [], set()

        def topo(u):                       # topological sort of the graph
            if id(u) not in seen:
                seen.add(id(u))
                for par, _ in u.parents:
                    topo(par)
                order.append(u)
        topo(self)
        self.grad = 1.0                    # seed: d f / d f = 1
        for u in reversed(order):
            for par, local in u.parents:
                par.grad += local * u.grad  # adjoint accumulation


def rlog(a):
    return Var(math.log(a.v), ((a, 1.0 / a.v),))


def rsin(a):
    return Var(math.sin(a.v), ((a, math.cos(a.v)),))


# f(x1, x2) = log(x1) + x1 * x2 - sin(x2)  at (2, 5)
f_fwd = lambda a, b: dlog(a) + a * b - dsin(b)
f_rev = lambda a, b: rlog(a) + a * b - rsin(b)
x1, x2 = 2.0, 5.0

g1 = f_fwd(Dual(x1, 1.0), Dual(x2, 0.0)).d   # pass 1: seed e1
g2 = f_fwd(Dual(x1, 0.0), Dual(x2, 1.0)).d   # pass 2: seed e2
print(f"forward mode (2 passes): grad = ({g1:.6f}, {g2:.6f})")

a, b = Var(x1), Var(x2)
out = f_rev(a, b)
out.backward()                               # one reverse sweep
print(f"reverse mode (1 sweep) : grad = ({a.grad:.6f}, {b.grad:.6f})")
print(f"analytic               : grad = ({1 / x1 + x2:.6f}, "
      f"{x1 - math.cos(x2):.6f})")
print(f"function value         : f = {out.v:.6f}")
