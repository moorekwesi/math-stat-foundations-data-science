# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Regularised PMT by CV; inclusion/exclusion errors at a budget; ROC; fairness.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.preprocessing import StandardScaler


def simulate_households(rng, n=4000, n_noise=30):
    """Simulated survey (as in ch13_pmt_logit.py) plus irrelevant items."""
    rural = (rng.random(n) < 0.55).astype(float)
    hhsize = np.clip(rng.poisson(4.5 + 1.0 * rural), 1, 14).astype(float)
    welfare = rng.standard_normal(n) - 0.6 * rural
    a = np.array([0.8, -0.2, -1.0, 1.2, -0.8, 0.0, 0.3, -0.4, -0.3, -1.3])
    b = np.array([0.5, 1.1, 1.2, 0.6, 0.6, 0.9, 1.0, 0.9, 1.0, 1.1])
    assets = (a + np.outer(welfare, b)
              + rng.standard_normal((n, a.size)) > 0).astype(float)
    logc = 7.0 + 0.55 * welfare - 0.07 * hhsize + 0.35 * rng.standard_normal(n)
    poor = (logc < np.quantile(logc, 0.30)).astype(float)
    noise = (rng.random((n, n_noise)) < 0.4).astype(float)
    return assets, hhsize, rural, poor, noise


rng = np.random.default_rng(803)
assets, hhsize, rural, poor, noise = simulate_households(rng)
n = poor.size
train = rng.random(n) < 0.25                        # small training sample
test = ~train
X = np.column_stack([assets, hhsize, rural, noise])
sc = StandardScaler().fit(X[train])
Xs = sc.transform(X)
ytr, yte = poor[train], poor[test]

# PCA asset index fitted on the training sample
Za = (assets - assets[train].mean(0)) / assets[train].std(0)
w1 = np.linalg.svd(Za[train], full_matrices=False)[2][0]
pca_score = -Za @ (w1 * np.sign(w1.sum()))          # high = predicted poor

Cs = np.logspace(-3, 1, 25)
models = {
    "logit (no penalty)": LogisticRegression(penalty="none", max_iter=5000),
    "logit L2, CV": LogisticRegressionCV(Cs=Cs, cv=5, penalty="l2",
                                         scoring="neg_log_loss", max_iter=5000),
    "logit L1, CV": LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1",
                                         solver="liblinear",
                                         scoring="neg_log_loss"),
}
scores = {"PCA asset index": pca_score[test]}
for name, m in models.items():
    m.fit(Xs[train], ytr)
    scores[name] = m.predict_proba(Xs[test])[:, 1]
l1 = models["logit L1, CV"]
print(f"CV-chosen C: L2 {models['logit L2, CV'].C_[0]:.3f}, L1 {l1.C_[0]:.3f}; "
      f"L1 keeps {np.sum(l1.coef_ != 0)} of {X.shape[1]} predictors "
      f"({np.sum(l1.coef_[0, 12:] != 0)} of 30 noise items)")

share = 0.25                                         # budget covers 25%
print(f"\nbudget covers {share:.0%} of households; poverty rate "
      f"{yte.mean():.1%} (test)")
print(f"{'method':<20}{'AUC':>7}{'logloss':>9}{'inclusion':>11}{'exclusion':>11}")
for name, s in scores.items():
    sel = s >= np.quantile(s, 1 - share)
    incl = np.mean(yte[sel] == 0)                    # benefits to non-poor
    excl = np.mean(sel[yte == 1] == 0)               # poor left out
    ll = "--" if name.startswith("PCA") else f"{log_loss(yte, s):.3f}"
    print(f"{name:<20}{roc_auc_score(yte, s):7.3f}{ll:>9}{incl:11.1%}"
          f"{excl:11.1%}")

# Fairness audit of the L2 model: error rates by area
s = scores["logit L2, CV"]
sel = s >= np.quantile(s, 1 - share)
print("\nL2 PMT by area      poverty  coverage  exclusion  inclusion")
for lab, g in [("urban", rural[test] == 0), ("rural", rural[test] == 1)]:
    pr, cov = yte[g].mean(), sel[g].mean()
    ex = np.mean(sel[g & (yte == 1)] == 0)
    inc = np.mean(yte[g & sel] == 0)
    print(f"  {lab:<17}{pr:9.1%}{cov:10.1%}{ex:11.1%}{inc:11.1%}")

fig, ax = plt.subplots(figsize=(6, 3.8))
for name in ["PCA asset index", "logit L2, CV"]:
    fpr, tpr, _ = roc_curve(yte, scores[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, scores[name]):.3f})")
    sel = scores[name] >= np.quantile(scores[name], 1 - share)
    ax.plot(np.mean(sel[yte == 0]), np.mean(sel[yte == 1]), "ko", ms=5)
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("false positive rate (non-poor selected)")
ax.set_ylabel("true positive rate (poor selected)")
ax.set_title("ROC curves; dots = 25% budget operating point")
ax.legend(loc="lower right", fontsize=9)
plt.savefig("ch13_pmt_roc.pdf", bbox_inches="tight")
