# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Monty Hall problem: simulate sticking versus switching with k doors.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)


def play(k, reps):
    """Host opens all but one of the unchosen doors, never revealing the car."""
    car = rng.integers(0, k, reps)
    pick = rng.integers(0, k, reps)
    # remaining closed door: the car if pick != car, else a random goat door
    other = rng.integers(0, k - 1, reps)
    other = other + (other >= pick)         # uniform over doors != pick
    remaining = np.where(pick != car, car, other)
    return np.mean(pick == car), np.mean(remaining == car)


print(f"{'doors':>5} {'stick sim':>10} {'stick exact':>12} "
      f"{'switch sim':>11} {'switch exact':>13}")
for k in [3, 4, 10, 100]:
    stick, switch = play(k, 200_000)
    print(f"{k:>5} {stick:10.4f} {1 / k:12.4f} {switch:11.4f} {1 - 1 / k:13.4f}")
