"""Evaluation protocol: repeated stratified k-fold cross-validation.

* stratified on (label x sensitive group) so every fold keeps the same group/class mix
* all preprocessing and mitigation fitted on training folds only
* metrics computed on the pooled out-of-fold predictions of each repeat
* results reported as mean +/- SD over repeats
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import SEED
from .fairness import fairness_report
from .mitigation import BASELINE, fit_predict


def cross_validate(X, y, A, model_name, method=BASELINE, n_splits=5, n_repeats=5, threshold=0.5,
                   proxy_cols=(), train_filter=None, return_predictions=False, weights=None):
    """weights: optional survey weights; if given, extra weighted metrics are added with the prefix 'W '."""
    y, A = pd.Series(np.asarray(y)), pd.Series(np.asarray(A))
    key = A.astype(str) + "_" + y.astype(str)
    # tiny strata (< n_splits members) are merged into their label stratum
    small = key.map(key.value_counts()) < n_splits
    key[small] = "rare_" + y[small].astype(str)
    rows, preds = [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        pred, proba = np.zeros(len(y), int), np.full(len(y), np.nan)
        rng = np.random.default_rng(SEED + rep)
        for tr, te in skf.split(X, key):
            if train_filter is not None:
                tr = train_filter(tr, y, A, rng)
            p, pr = fit_predict(method, model_name, X.iloc[tr], y.iloc[tr], A.iloc[tr], X.iloc[te], A.iloc[te],
                                threshold=threshold, proxy_cols=proxy_cols, seed=SEED + rep)
            pred[te] = p
            if pr is not None:
                proba[te] = pr
        pr_ok = None if np.isnan(proba).any() else proba
        row = fairness_report(y, pred, A, pr_ok)
        if weights is not None:
            wr = fairness_report(y, pred, A, pr_ok, w=np.asarray(weights, float))
            row.update({f"W {k}": v for k, v in wr.items() if "[" not in k or k.startswith("TPR [")})
        rows.append(row)
        preds.append((pred, proba.copy()))
    res = pd.DataFrame(rows)
    return (res, preds) if return_predictions else res


def summarize(cv: pd.DataFrame, cols) -> pd.Series:
    return pd.Series({c: (f"{cv[c].mean():.3f} ± {cv[c].std(ddof=1) if len(cv) > 1 else 0:.3f}"
                          if c in cv and cv[c].notna().all() else "—") for c in cols})


def downsample_group(group, fraction):
    """Training-data filter: keep only `fraction` of one group's patients (simulates under-representation)."""
    def f(tr, y, A, rng):
        tr = np.asarray(tr)
        if fraction >= 1:
            return tr
        g = tr[np.asarray(A)[tr] == group]
        keep = [tr[np.asarray(A)[tr] != group]]
        for c in (0, 1):
            idx = g[np.asarray(y)[g] == c]
            if len(idx):
                keep.append(rng.choice(idx, size=min(len(idx), max(2, round(fraction * len(idx)))), replace=False))
        return np.concatenate(keep)
    return f


def bootstrap_compare(y, A, pred_a, pred_b, n_boot=1000, seed=0, w=None):
    """Paired bootstrap over patients: difference (b - a) in Equalized Odds gap and balanced accuracy.
    Returns mean difference, 95% CI and a two-sided bootstrap p-value for each."""
    y, A, pa, pb = (np.asarray(v) for v in (y, A, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    d_eo, d_ba = [], []
    groups = np.unique(A)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(A[idx])) < len(groups):
            continue
        ra = fairness_report(y[idx], pa[idx], A[idx], w=None if w is None else np.asarray(w)[idx])
        rb = fairness_report(y[idx], pb[idx], A[idx], w=None if w is None else np.asarray(w)[idx])
        d_eo.append(rb["EOdds diff"] - ra["EOdds diff"]); d_ba.append(rb["Balanced acc."] - ra["Balanced acc."])
    out = {}
    for name, d in [("EOdds diff", np.array(d_eo)), ("Balanced acc.", np.array(d_ba))]:
        d = d[~np.isnan(d)]
        p = 2 * min((d >= 0).mean(), (d <= 0).mean())
        out[name] = {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                     "p": float(min(1.0, p))}
    return out
