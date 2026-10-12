# DEP 803 -- Chapter 0: Mathematical Refresher
# Lab 0, part 3: what common error messages look like, and what they mean.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

examples = {
    "NameError": "print(revenu)",
    "TypeError": "'Total: ' + 250",
    "IndexError": "[10, 20, 30][3]",
    "ZeroDivisionError": "1000 / 0",
    "ValueError": "np.ones(3) @ np.ones(4)",
    "SyntaxError": "print('hello'",
}
revenue = 250
for expected, code in examples.items():
    try:
        eval(compile(code, "<lab0>", "eval"))
    except Exception as err:   # catch the error so the script keeps running
        msg = str(err).splitlines()[0]
        msg = msg if len(msg) <= 40 else msg[:37] + "..."
        print(f"{code:24s} -> {type(err).__name__}: {msg}")

# NumPy is quieter: division by zero gives inf or nan with a warning
with np.errstate(divide="ignore", invalid="ignore"):
    print("np.array([1.0, 0.0]) / 0 =", np.array([1.0, 0.0]) / 0)

# getting help: every function carries its own documentation
print("first line of help(np.mean):", np.mean.__doc__.strip().splitlines()[0])
