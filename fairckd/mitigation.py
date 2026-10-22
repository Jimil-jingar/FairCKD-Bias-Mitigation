"""Stage 7 - Bias reduction.

| Method                         | Type             | Reference                    |
|--------------------------------|------------------|------------------------------|
| Unawareness                    | naive baseline   | drop the sensitive attribute |
| Reweighing                     | pre-processing   | Kamiran & Calders, 2012      |
| Group-balanced SMOTE           | pre-processing   | Chawla et al., 2002 (per group x label cell) |
| Exponentiated Gradient + EOdds | in-processing    | Agarwal et al., 2018         |
| Adversarial Debiasing + EOdds  | in-processing    | Zhang, Lemoine & Mitchell, 2018 (Logistic Regression predictor) |
| Threshold Optimizer + EOdds    | post-processing  | Hardt et al., 2016           |
| Reject Option Classification   | post-processing  | Kamiran, Karim & Zhang, 2012 |
"""
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ErrorRate, ExponentiatedGradient
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import SEED
from .adversarial import fit_tuned as fit_adversarial
from .fairness import reweighing_weights
from .models import Float64Proba, make_classifier
from .preprocessing import make_preprocessor

BASELINE = "Baseline"
UNAWARE = "Unawareness (drop sensitive attribute)"
REWEIGH = "Reweighing (pre-processing)"
EXPGRAD = "Exponentiated Gradient (in-processing)"
THRESH = "Threshold Optimizer (post-processing)"
HYBRID = "Reweighing + Threshold Optimizer (hybrid)"
SMOTE = "Group-balanced SMOTE (pre-processing)"
REJECT = "Reject Option Classification (post-processing)"
ADVERSARIAL = "Adversarial Debiasing (in-processing)"
METHODS = [BASELINE, UNAWARE, REWEIGH, SMOTE, EXPGRAD, ADVERSARIAL, THRESH, REJECT, HYBRID]
SHORT = {BASELINE: "Baseline", UNAWARE: "Unawareness", REWEIGH: "Reweighing", SMOTE: "SMOTE",
         EXPGRAD: "Exp. Gradient", ADVERSARIAL: "Adversarial", THRESH: "Threshold Opt.", REJECT: "Reject Option",
         HYBRID: "RW + TO (hybrid)"}
# short name -> full name (saved results store short names in some tables)
LONG = {v: k for k, v in SHORT.items()}


def supports(method, model_name) -> bool:
    """Whether a mitigation method can be combined with a classifier.
    * Exponentiated Gradient needs sample weights (KNN has none).
    * Adversarial Debiasing trains its own logistic-regression predictor, so it is paired with Logistic Regression only."""
    if method == EXPGRAD and model_name == "KNN":
        return False
    if method == ADVERSARIAL and model_name != "Logistic Regression":
        return False
    return True


def _cut(proba, ytr, threshold):
    t = float(np.mean(ytr)) if threshold == "prevalence" else float(threshold)
    return (np.asarray(proba) >= t).astype(int)


def group_smote(Z, y, A, k=5, seed=SEED):
    """Oversample every (group, label) cell to the size of the largest cell with SMOTE interpolation.

    Plain SMOTE only balances the classes; here each demographic group gets as many CKD and non-CKD
    examples as the best-represented cell, so no group is under-represented in either class.
    """
    rng = np.random.default_rng(seed)
    Z, y, A = np.asarray(Z, dtype=float), np.asarray(y), np.asarray(A)
    cells = {(a, c): np.where((A == a) & (y == c))[0] for a in np.unique(A) for c in (0, 1)}
    target = max(len(i) for i in cells.values())
    newZ, newy = [Z], [y]
    for (a, c), idx in cells.items():
        need = target - len(idx)
        if need <= 0 or len(idx) == 0:
            continue
        P = Z[idx]
        if len(idx) == 1:                                # nothing to interpolate with: duplicate
            newZ.append(np.repeat(P, need, axis=0)); newy.append(np.full(need, c)); continue
        kk = min(k, len(idx) - 1)
        d = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :kk]
        base = rng.integers(0, len(idx), need)
        mate = nn[base, rng.integers(0, kk, need)]
        gap = rng.random((need, 1))
        newZ.append(P[base] + gap * (P[mate] - P[base])); newy.append(np.full(need, c))
    return np.vstack(newZ), np.concatenate(newy)


def _eodds_gap(y, pred, A):
    gaps = []
    for lab in (1, 0):
        rates = [pred[(A == a) & (y == lab)].mean() for a in np.unique(A) if ((A == a) & (y == lab)).any()]
        gaps.append(max(rates) - min(rates) if rates else 0.0)
    return max(gaps)


def _bal_acc(y, pred):
    tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
    tnr = 1 - pred[y == 0].mean() if (y == 0).any() else 0.0
    return (tpr + tnr) / 2


def reject_option(p, A, t, low_groups, theta):
    """Inside the band |p - t| <= theta, the groups with lower detection get a CKD flag and the others do not."""
    p, A = np.asarray(p), np.asarray(A)
    pred = (p >= t).astype(int)
    band = np.abs(p - t) <= theta
    low = np.isin(A, list(low_groups))
    pred[band & low] = 1
    pred[band & ~low] = 0
    return pred


def fit_reject_option(oof, y, A, t, max_loss=0.03):
    """Learn the disadvantaged groups and the band width on out-of-fold training scores.

    The widest fairness gain is chosen among bands that keep balanced accuracy within `max_loss`.
    """
    oof, y, A = np.asarray(oof), np.asarray(y), np.asarray(A)
    base = (oof >= t).astype(int)
    overall = base[y == 1].mean()
    low = [a for a in np.unique(A) if ((A == a) & (y == 1)).any() and base[(A == a) & (y == 1)].mean() < overall]
    best_theta, best_gap = 0.0, _eodds_gap(y, base, A)
    ref = _bal_acc(y, base)
    for theta in np.linspace(0.01, 0.4, 40):
        pr = reject_option(oof, A, t, low, theta)
        if _bal_acc(y, pr) < ref - max_loss:
            continue
        g = _eodds_gap(y, pr, A)
        if g < best_gap - 1e-9:
            best_theta, best_gap = float(theta), g
    return low, best_theta


def fit_predict(method, model_name, Xtr, ytr, Atr, Xte, Ate, threshold=0.5, proxy_cols=(), seed=SEED):
    """Train `model_name` with `method` on the training fold; return (predictions, probabilities|None)."""
    ytr, Atr, Ate = np.asarray(ytr), np.asarray(Atr), np.asarray(Ate)
    if method == UNAWARE:
        drop = [c for c in proxy_cols if c in Xtr.columns]
        Xtr, Xte = Xtr.drop(columns=drop), Xte.drop(columns=drop)
    pre = make_preprocessor().fit(Xtr)
    Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
    clf = make_classifier(model_name)

    if method in (BASELINE, UNAWARE):
        clf.fit(Ztr, ytr)
        p = clf.predict_proba(Zte)[:, 1]
        return _cut(p, ytr, threshold), p

    if method == REWEIGH:
        w = reweighing_weights(ytr, Atr)
        if model_name == "KNN":                      # no sample_weight support -> weighted resampling
            idx = np.random.default_rng(seed).choice(len(ytr), len(ytr), replace=True, p=w / w.sum())
            clf.fit(Ztr[idx], ytr[idx])
        else:
            clf.fit(Ztr, ytr, sample_weight=w)
        p = clf.predict_proba(Zte)[:, 1]
        return _cut(p, ytr, threshold), p

    if method == SMOTE:
        Zs, ys = group_smote(Ztr, ytr, Atr, seed=seed)
        clf.fit(Zs, ys)
        p = clf.predict_proba(Zte)[:, 1]
        # the oversampled training set is already class-balanced, so the "prevalence" rule becomes 0.5
        return _cut(p, ys, 0.5 if threshold == "prevalence" else threshold), p

    if method == REJECT:
        t = float(np.mean(ytr)) if threshold == "prevalence" else float(threshold)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = cross_val_predict(make_classifier(model_name), Ztr, ytr, cv=inner, method="predict_proba")[:, 1]
        low, theta = fit_reject_option(oof, ytr, Atr, t)
        clf.fit(Ztr, ytr)
        p = clf.predict_proba(Zte)[:, 1]
        return reject_option(p, Ate, t, low, theta), p

    if method == ADVERSARIAL:
        if model_name != "Logistic Regression":
            raise ValueError("Adversarial Debiasing uses a logistic-regression predictor; pair it with Logistic Regression")
        # alpha (adversary strength) is tuned on a validation split of the training fold only
        adv, _ = fit_adversarial(Ztr, ytr, Atr, cut=lambda pr, yt: _cut(pr, yt, threshold), seed=seed)
        p = adv.predict_proba(Zte)[:, 1]
        return _cut(p, ytr, threshold), p

    if method == EXPGRAD:
        if model_name == "KNN":
            raise ValueError("Exponentiated Gradient needs sample weights (not supported by KNN)")
        if threshold == "prevalence":
            # imbalanced data: optimise the BALANCED error (cost of a missed CKD case = 1/prevalence),
            # otherwise the constrained learner becomes "fair" by predicting almost nobody as CKD
            pos = float(np.mean(ytr))
            obj = ErrorRate(costs={"fn": 0.5 / pos, "fp": 0.5 / (1 - pos)})
            mit = ExponentiatedGradient(clf, constraints=EqualizedOdds(difference_bound=0.02), objective=obj, max_iter=50)
            mit.fit(Ztr, ytr, sensitive_features=Atr)
            p = mit._pmf_predict(Zte)[:, 1]
            return (p >= 0.5).astype(int), p
        mit = ExponentiatedGradient(clf, constraints=EqualizedOdds(difference_bound=0.02), max_iter=50)
        mit.fit(Ztr, ytr, sensitive_features=Atr)
        p = mit._pmf_predict(Zte)[:, 1]
        return _cut(p, ytr, threshold), p

    if method in (THRESH, HYBRID):
        # HYBRID (FairCKD): reweighing makes the training data group/label independent (pre-processing);
        # the threshold optimizer then equalises error rates on the reweighed model's scores (post-processing).
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_kw = {}
        if method == HYBRID and model_name != "KNN":
            fit_kw = {"sample_weight": reweighing_weights(ytr, Atr)}
        oof = cross_val_predict(make_classifier(model_name), Ztr, ytr, cv=inner, method="predict_proba",
                                params=fit_kw or None)
        clf.fit(Ztr, ytr, **fit_kw)
        to = ThresholdOptimizer(estimator=Float64Proba(clf, train_X=Ztr, train_proba=oof),
                                constraints="equalized_odds", objective="balanced_accuracy_score",
                                predict_method="predict_proba", prefit=True)
        to.fit(Ztr, ytr, sensitive_features=Atr)
        return np.asarray(to.predict(Zte, sensitive_features=Ate, random_state=seed)), None

    raise ValueError(method)
