"""Stages 5 and 6 - Bias detection (group-wise error rates) and fairness evaluation.
Works for two or more groups: every 'difference' metric is (max over groups - min over groups)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

METRIC_INFO = {
    "DP diff": ("Demographic Parity difference", "Gap in the share of patients flagged as CKD. 0 = equal. "
                "Not a suitable target when true CKD prevalence differs between groups."),
    "DI ratio": ("Disparate Impact ratio", "Lowest / highest flag rate. 1 = equal; < 0.8 fails the '80 % rule'."),
    "EOpp diff": ("Equal Opportunity difference", "Gap in sensitivity (share of real CKD patients detected). "
                  "0 = every group's CKD patients are detected equally well."),
    "EOdds diff": ("Equalized Odds difference", "Largest gap in sensitivity OR false-positive rate. 0 = equal error rates."),
    "PP diff": ("Predictive Parity difference", "Gap in precision (how often a CKD flag is correct)."),
    "Acc. gap": ("Accuracy gap", "Gap in accuracy between groups."),
}
PERF = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Balanced acc."]
FAIR = list(METRIC_INFO)


def _rates(y, p, w=None):
    """Group rates; optional sample weights w (e.g. NHANES survey weights -> population estimates)."""
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    tp = w[(y == 1) & (p == 1)].sum(); fn = w[(y == 1) & (p == 0)].sum()
    fp = w[(y == 0) & (p == 1)].sum(); tn = w[(y == 0) & (p == 0)].sum()
    n = w.sum()
    div = lambda a, b: a / b if b else np.nan
    return {"n": len(y), "CKD cases": int(((y == 1)).sum()), "Selection rate": div(tp + fp, n),
            "TPR": div(tp, tp + fn), "FPR": div(fp, fp + tn), "FNR": div(fn, tp + fn),
            "PPV": div(tp, tp + fp), "Accuracy": div(tp + tn, n)}


def group_rates(y, pred, A, w=None) -> pd.DataFrame:
    y, pred, A = np.asarray(y), np.asarray(pred), np.asarray(A)
    w = None if w is None else np.asarray(w, float)
    return pd.DataFrame({g: _rates(y[A == g], pred[A == g], None if w is None else w[A == g])
                         for g in sorted(np.unique(A))}).T


def _gap(s):
    s = s.dropna()
    return float(s.max() - s.min()) if len(s) > 1 else 0.0


def fairness_report(y, pred, A, proba=None, w=None) -> dict:
    """Performance + fairness metrics. w: optional sample (survey) weights."""
    y, pred = np.asarray(y), np.asarray(pred)
    r = group_rates(y, pred, A, w)
    sr = r["Selection rate"]
    tot = _rates(y, pred, w)
    rec = tot["TPR"]; spec_ = 1 - tot["FPR"] if not np.isnan(tot["FPR"]) else np.nan
    prec = tot["PPV"] if not np.isnan(tot["PPV"]) else 0.0
    out = {
        "Accuracy": tot["Accuracy"], "Precision": prec, "Recall": rec,
        "F1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
        "AUC": roc_auc_score(y, proba, sample_weight=w) if proba is not None and len(np.unique(y)) > 1 else np.nan,
        "Balanced acc.": 0.5 * (rec + spec_),
        "DP diff": _gap(sr), "DI ratio": float(sr.min() / sr.max()) if sr.max() > 0 else np.nan,
        "EOpp diff": _gap(r["TPR"]), "EOdds diff": max(_gap(r["TPR"]), _gap(r["FPR"])),
        "PP diff": _gap(r["PPV"]), "Acc. gap": _gap(r["Accuracy"]),
    }
    for g in r.index:
        for k in ["TPR", "FPR", "PPV", "Accuracy", "Selection rate"]:
            out[f"{k} [{g}]"] = r.loc[g, k]
    return out


def calibration_by_group(y, proba, A, bins=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group calibration: calibration-in-the-large (mean predicted - observed), expected calibration
    error (ECE) and Brier score, plus reliability-curve points for plotting."""
    y, proba, A = np.asarray(y), np.asarray(proba, float), np.asarray(A)
    rows, curve = [], []
    edges = np.linspace(0, 1, bins + 1)
    for g in sorted(np.unique(A)):
        m = A == g
        yy, pp = y[m], proba[m]
        idx = np.clip(np.digitize(pp, edges[1:-1]), 0, bins - 1)
        ece = 0.0
        for b in range(bins):
            k = idx == b
            if k.any():
                ece += k.mean() * abs(pp[k].mean() - yy[k].mean())
                curve.append({"Group": g, "Mean predicted": pp[k].mean(), "Observed CKD rate": yy[k].mean(), "n": int(k.sum())})
        rows.append({"Group": g, "Patients": int(m.sum()), "Observed CKD (%)": 100 * yy.mean(),
                     "Mean predicted (%)": 100 * pp.mean(), "Calibration-in-the-large (pp)": 100 * (pp.mean() - yy.mean()),
                     "ECE": ece, "Brier": float(np.mean((pp - yy) ** 2))})
    return pd.DataFrame(rows).round(4), pd.DataFrame(curve)


def reference_values(y, A) -> dict:
    """Metric values of a PERFECT classifier: shows which metrics are driven by base rates."""
    return fairness_report(y, y, A)


def reweighing_weights(y, A) -> np.ndarray:
    """Kamiran & Calders (2012): w(a, y) = P(a) P(y) / P(a, y)."""
    y, A = np.asarray(y), np.asarray(A)
    w = np.ones(len(y))
    for a in np.unique(A):
        for c in np.unique(y):
            m = (A == a) & (y == c)
            if m.any():
                w[m] = np.mean(A == a) * np.mean(y == c) / np.mean(m)
    return w
