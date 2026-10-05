# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Naive Bayes poverty classifier built from counts (simulated survey).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.naive_bayes import BernoulliNB

rng = np.random.default_rng(803)
n = 20_000
poor = rng.random(n) < 0.3
names = ["thatch roof", "no phone", "no toilet", "size > 6"]
p_given_poor = np.array([0.55, 0.45, 0.60, 0.40])      # P(x_j = 1 | poor)
p_given_rich = np.array([0.15, 0.10, 0.20, 0.15])      # P(x_j = 1 | non-poor)
probs = np.where(poor[:, None], p_given_poor, p_given_rich)
X = (rng.random((n, 4)) < probs).astype(int)
train, test = slice(0, 15_000), slice(15_000, n)

# "Fitting" = estimating a prior and four conditional probabilities per class
yt, Xt = poor[train], X[train]
prior = yt.mean()
th1 = (Xt[yt].sum(0) + 1) / (yt.sum() + 2)             # Laplace smoothing
th0 = (Xt[~yt].sum(0) + 1) / ((~yt).sum() + 2)
print(f"Estimated prior P(poor) = {prior:.3f}")
for j, nm in enumerate(names):
    print(f"  P({nm:<11} | poor) = {th1[j]:.3f}   | non-poor) = {th0[j]:.3f}")


def posterior(Xn):
    l1 = np.log(prior) + (Xn * np.log(th1) + (1 - Xn) * np.log(1 - th1)).sum(1)
    l0 = np.log(1 - prior) + (Xn * np.log(th0)
                              + (1 - Xn) * np.log(1 - th0)).sum(1)
    return 1 / (1 + np.exp(l0 - l1))


post = posterior(X[test])
acc = np.mean((post > 0.5) == poor[test])
sk = BernoulliNB(alpha=1.0).fit(Xt, yt)
post_sk = sk.predict_proba(X[test])[:, 1]
print(f"\nTest accuracy (threshold 0.5)         = {acc:.3f}")
print(f"Max |ours - sklearn BernoulliNB|      = {np.max(abs(post - post_sk)):.2e}")
print("Posterior P(poor | all four deprivations) = "
      f"{posterior(np.ones((1, 4)))[0]:.3f}")
print("Posterior P(poor | none of them)          = "
      f"{posterior(np.zeros((1, 4)))[0]:.3f}")
