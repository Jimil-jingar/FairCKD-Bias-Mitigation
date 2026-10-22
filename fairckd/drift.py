"""Monitoring - data drift and fairness drift between two cohorts.

A model that was fair when it was validated can become unfair later, because the patient mix, the measurements
or the disease prevalence change. FairCKD checks this with the two NHANES cycles (development 2017-18,
later-deployment proxy 2015-16 - a different sample of the same population):

  * Data drift: Population Stability Index (PSI) of every feature, overall and inside each group.
    PSI < 0.1 = stable, 0.1-0.2 = moderate shift, > 0.2 = significant shift (common credit-risk rule of thumb).
  * Prevalence drift: CKD base rate per group in each cohort.
  * Fairness drift: change of each pipeline's Equalized Odds gap and of each group's sensitivity between the
    internal cross-validated estimate and the external cohort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE, PSI_MAJOR = 0.1, 0.2


def psi(ref, new, bins=10) -> float:
    """Population Stability Index of `new` against the reference distribution `ref` (quantile bins of ref)."""
    ref = np.asarray(ref, float); new = np.asarray(new, float)
    ref, new = ref[~np.isnan(ref)], new[~np.isnan(new)]
    if len(ref) < 10 or len(new) < 10:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # (almost) binary feature: compare category shares
        cats = np.unique(np.concatenate([ref, new]))
        r = np.array([(ref == c).mean() for c in cats]); n = np.array([(new == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref); n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def _level(v):
    if np.isnan(v):
        return "n/a"
    return "Significant" if v > PSI_MAJOR else "Moderate" if v > PSI_MODERATE else "Stable"


def data_drift(X_ref, A_ref, X_new, A_new, labels=None) -> pd.DataFrame:
    labels = labels or {}
    A_ref, A_new = np.asarray(A_ref).astype(str), np.asarray(A_new).astype(str)
    groups = sorted(set(A_ref) & set(A_new))
    rows = []
    for c in X_ref.columns:
        row = {"Feature": labels.get(c, c), "PSI (all)": psi(X_ref[c], X_new[c])}
        for g in groups:
            row[f"PSI [{g}]"] = psi(X_ref[c].values[A_ref == g], X_new[c].values[A_new == g])
        gv = [row[f"PSI [{g}]"] for g in groups]
        row["Max group PSI"] = np.nanmax(gv) if not np.all(np.isnan(gv)) else np.nan
        row["Shift"] = _level(row["Max group PSI"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Max group PSI", ascending=False).round(3).reset_index(drop=True)


def prevalence_drift(y_ref, A_ref, y_new, A_new) -> pd.DataFrame:
    ref = pd.DataFrame({"y": np.asarray(y_ref), "A": np.asarray(A_ref).astype(str)})
    new = pd.DataFrame({"y": np.asarray(y_new), "A": np.asarray(A_new).astype(str)})
    out = pd.DataFrame({"Patients (development)": ref.groupby("A").size(),
                        "CKD % (development)": 100 * ref.groupby("A").y.mean(),
                        "Patients (external)": new.groupby("A").size(),
                        "CKD % (external)": 100 * new.groupby("A").y.mean()})
    out["Change (pp)"] = out["CKD % (external)"] - out["CKD % (development)"]
    out.index.name = "Group"
    return out.round(2).reset_index()


def fairness_drift(mitigation_raw: pd.DataFrame, external: pd.DataFrame, short: dict) -> pd.DataFrame:
    """Internal (cross-validated) vs external Equalized Odds gap and balanced accuracy for every pipeline."""
    internal = mitigation_raw.assign(Method=mitigation_raw.Method.map(lambda m: short.get(m, m))) \
        .groupby(["Model", "Method"], sort=False)[["Balanced acc.", "EOdds diff"]].mean()
    ext = external.set_index(["Model", "Method"])[["Balanced acc.", "EOdds diff"]]
    d = internal.join(ext, lsuffix=" (internal)", rsuffix=" (external)", how="inner")
    d["Δ EOdds gap"] = d["EOdds diff (external)"] - d["EOdds diff (internal)"]
    d["Δ Balanced acc."] = d["Balanced acc. (external)"] - d["Balanced acc. (internal)"]
    d["Fairness held?"] = np.where(d["Δ EOdds gap"] <= 0.05, "Yes", "No (gap grew > 0.05)")
    return d.round(3).reset_index()


def sensitivity_drift(baseline_raw: pd.DataFrame, external: pd.DataFrame, groups) -> pd.DataFrame:
    """Per-group sensitivity (TPR) of the unmitigated models: internal CV vs external cohort."""
    rows = []
    ext = external[external.Method == "Baseline"].set_index("Model")
    bm = baseline_raw.groupby("Model").mean(numeric_only=True)
    for m in ext.index:
        if m not in bm.index:
            continue
        for g in groups:
            c = f"TPR [{g}]"
            if c in bm.columns and c in ext.columns:
                rows.append({"Model": m, "Group": g, "Sensitivity (internal)": bm.loc[m, c],
                             "Sensitivity (external)": ext.loc[m, c], "Change": ext.loc[m, c] - bm.loc[m, c]})
    return pd.DataFrame(rows).round(3)
