"""FairCKD-Select: fairness-constrained model and mitigation selection (proposed method).

Problem. No single bias-mitigation method is best for every patient group (see the mitigation
experiments): threshold optimisation wins for age on UCI, exponentiated gradient for race on NHANES,
reweighing is almost free when gaps are moderate, and every method has a different accuracy cost.
Choosing the method by looking at test results would be optimistic and unfair to the baseline.

Method. For every candidate pipeline c = (classifier, mitigation method):
  1. inner K-fold cross-validation on the TRAINING data only gives the expected balanced accuracy BA(c)
     and Equalized Odds gap EO(c);
  2. the accuracy reference is the best baseline (unmitigated) balanced accuracy BA*;
  3. feasible set F = { c : BA(c) >= BA* - delta }   (clinical utility tolerance, default delta = 0.03);
  4. FairCKD-Select picks  c* = argmin_{c in F} EO(c)  (ties: higher BA);
  5. c* is refitted on the full training data and evaluated once on held-out data.
Steps 1-5 are wrapped in an outer cross-validation (nested CV), so the reported performance of the
selected pipeline is unbiased. The same nested procedure with the rule "highest BA" gives the
conventional accuracy-driven choice used for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import SEED
from .fairness import fairness_report
from .mitigation import BASELINE, EXPGRAD, HYBRID, REWEIGH, SHORT, THRESH, UNAWARE, fit_predict

DEFAULT_METHODS = [BASELINE, UNAWARE, REWEIGH, EXPGRAD, THRESH, HYBRID]


def _key(y, A):
    k = pd.Series(np.asarray(A)).astype(str) + "_" + pd.Series(np.asarray(y)).astype(str)
    small = k.map(k.value_counts()) < 3
    k[small] = "rare_" + pd.Series(np.asarray(y))[small].astype(str)
    return k


def select_rule(scores: pd.DataFrame, delta=0.03, rule="fair"):
    """scores: rows = candidates (index (model, method)), columns 'Balanced acc.', 'EOdds diff'."""
    if rule == "accuracy":
        base = scores[[m == BASELINE for _, m in scores.index]]
        return base["Balanced acc."].idxmax()
    base_ba = scores[[m == BASELINE for _, m in scores.index]]["Balanced acc."].max()
    feas = scores[scores["Balanced acc."] >= base_ba - delta]
    if feas.empty:
        feas = scores
    best = feas["EOdds diff"].min()
    tie = feas[feas["EOdds diff"] <= best + 1e-9]
    return tie["Balanced acc."].idxmax()


def inner_scores(X, y, A, candidates, threshold, proxy, inner_splits=3, seed=SEED):
    y, A = pd.Series(np.asarray(y)), pd.Series(np.asarray(A))
    skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    preds = {c: np.zeros(len(y), int) for c in candidates}
    for tr, te in skf.split(X, _key(y, A)):
        for c in candidates:
            p, _ = fit_predict(c[1], c[0], X.iloc[tr], y.iloc[tr], A.iloc[tr], X.iloc[te], A.iloc[te],
                               threshold=threshold, proxy_cols=proxy, seed=seed)
            preds[c][te] = p
    rows = {}
    for c, p in preds.items():
        r = fairness_report(y, p, A)
        rows[c] = {"Balanced acc.": r["Balanced acc."], "EOdds diff": r["EOdds diff"]}
    return pd.DataFrame(rows).T


def nested_select(X, y, A, models=("Logistic Regression", "XGBoost"), methods=DEFAULT_METHODS, threshold=0.5,
                  proxy_cols=(), delta=0.03, outer_splits=5, outer_repeats=2, inner_splits=3, log=print):
    """Nested-CV evaluation of FairCKD-Select vs. accuracy-driven selection.
    Returns (per-repeat metrics DataFrame, selection log DataFrame)."""
    y, A = pd.Series(np.asarray(y)), pd.Series(np.asarray(A))
    methods = [m for m in methods if not (m == UNAWARE and not proxy_cols)]
    candidates = [(mo, me) for mo in models for me in methods if not (me == EXPGRAD and mo == "KNN")]
    rows, picks = [], []
    for rep in range(outer_repeats):
        skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=SEED + 100 + rep)
        pred = {"FairCKD-Select": np.zeros(len(y), int), "Accuracy-driven selection": np.zeros(len(y), int)}
        for f, (tr, te) in enumerate(skf.split(X, _key(y, A))):
            sc = inner_scores(X.iloc[tr], y.iloc[tr], A.iloc[tr], candidates, threshold, proxy_cols,
                              inner_splits, seed=SEED + rep * 10 + f)
            for name, rule in [("FairCKD-Select", "fair"), ("Accuracy-driven selection", "accuracy")]:
                c = select_rule(sc, delta, rule)
                p, _ = fit_predict(c[1], c[0], X.iloc[tr], y.iloc[tr], A.iloc[tr], X.iloc[te], A.iloc[te],
                                   threshold=threshold, proxy_cols=proxy_cols, seed=SEED + rep)
                pred[name][te] = p
                picks.append({"Repeat": rep + 1, "Fold": f + 1, "Strategy": name, "Model": c[0],
                              "Method": SHORT[c[1]], "Inner balanced acc.": round(sc.loc[[c], "Balanced acc."].iloc[0], 3),
                              "Inner EOdds gap": round(sc.loc[[c], "EOdds diff"].iloc[0], 3)})
            log(f"      nested repeat {rep + 1} fold {f + 1}: FairCKD-Select -> "
                f"{picks[-2]['Model']} + {picks[-2]['Method']}")
        for name, p in pred.items():
            r = fairness_report(y, p, A); r["Strategy"] = name; r["Repeat"] = rep + 1
            rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(picks)


def pareto_front(points: pd.DataFrame, x="EOdds diff", y="Balanced acc.") -> pd.Series:
    """Boolean mask of Pareto-optimal points (lower x, higher y)."""
    px, py = points[x].values, points[y].values
    opt = []
    for i in range(len(points)):
        dominated = np.any((px <= px[i]) & (py >= py[i]) & ((px < px[i]) | (py > py[i])))
        opt.append(not dominated)
    return pd.Series(opt, index=points.index)
