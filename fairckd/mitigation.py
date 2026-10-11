"""Stage 7 - Bias reduction.

| Method                         | Type             | Reference                    |
|--------------------------------|------------------|------------------------------|
| Unawareness                    | naive baseline   | drop the sensitive attribute |
| Reweighing                     | pre-processing   | Kamiran & Calders, 2012      |
| Exponentiated Gradient + EOdds | in-processing    | Agarwal et al., 2018         |
| Threshold Optimizer + EOdds    | post-processing  | Hardt et al., 2016           |
"""
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ErrorRate, ExponentiatedGradient
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import SEED
from .fairness import reweighing_weights
from .models import Float64Proba, make_classifier
from .preprocessing import make_preprocessor

BASELINE = "Baseline"
UNAWARE = "Unawareness (drop sensitive attribute)"
REWEIGH = "Reweighing (pre-processing)"
EXPGRAD = "Exponentiated Gradient (in-processing)"
THRESH = "Threshold Optimizer (post-processing)"
HYBRID = "Reweighing + Threshold Optimizer (hybrid)"
METHODS = [BASELINE, UNAWARE, REWEIGH, EXPGRAD, THRESH, HYBRID]
SHORT = {BASELINE: "Baseline", UNAWARE: "Unawareness", REWEIGH: "Reweighing",
         EXPGRAD: "Exp. Gradient", THRESH: "Threshold Opt.", HYBRID: "RW + TO (hybrid)"}


def _cut(proba, ytr, threshold):
    t = float(np.mean(ytr)) if threshold == "prevalence" else float(threshold)
    return (np.asarray(proba) >= t).astype(int)


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
