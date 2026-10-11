"""End-to-end experiment runner: executes all nine framework stages for one
(dataset, feature set, sensitive attribute) configuration and stores every result."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import SEED, plots
from .data import ROOT, group_summary
from .diagnosis import train_final_model
from .evaluation import bootstrap_compare, cross_validate, downsample_group, summarize
from .explain import explain, importance_by_group
from .fairness import FAIR, PERF, calibration_by_group, fairness_report, reference_values
from .features import feature_ranking
from .mitigation import BASELINE, EXPGRAD, HYBRID, METHODS, REJECT, REWEIGH, SHORT, SMOTE, THRESH, UNAWARE, fit_predict
from .select import nested_select, pareto_front
from .models import MODEL_NAMES, make_classifier
from .preprocessing import make_preprocessor, missing_report

RESULTS = ROOT / "results"


def slug(spec, feature_set, attribute):
    fs = feature_set.split("(")[0].strip().lower().replace(" ", "-")
    return f"{spec.key}__{attribute.lower().replace('×', 'x').replace('/', '-').replace(' ', '-')}__{fs}"


def label_bias_table(spec) -> pd.DataFrame | None:
    if "ckd_2009" not in spec.df:
        return None
    g = spec.df.groupby("race")
    lb = pd.DataFrame({"Adults": g.size(),
                       "CKD % (2021 race-free eGFR)": 100 * g.ckd.mean(),
                       "CKD % (2009 race-based eGFR)": 100 * g.ckd_2009.mean()})
    lb["Change (percentage points)"] = lb.iloc[:, 2] - lb.iloc[:, 1]
    flips = spec.df.groupby("race").apply(lambda d: int((d.ckd != d.ckd_2009).sum()), include_groups=False)
    lb["Patients whose label changes"] = flips
    return lb.round(2)


def lime_agreement(model, pre, X, A, sv, spec, per_group=25, top=5, seed=SEED):
    """For a sample of patients per group: how many of the top-5 SHAP features LIME also ranks in its top 5."""
    from .explain import lime_explain
    rng = np.random.default_rng(seed)
    binary = [c for c in spec.binary if c in X.columns]
    rows = []
    Av = np.asarray(A)
    for g in sorted(pd.unique(Av)):
        idx = np.where(Av == g)[0]
        for i in rng.choice(idx, min(per_group, len(idx)), replace=False):
            lt, info = lime_explain(model, pre, X.iloc[[i]], spec.labels, binary, n_samples=1500, seed=int(i))
            s_top = {sv.feature_names[k] for k in np.argsort(-np.abs(sv[i].values))[:top]}
            rows.append({"Group": g, "Top-5 overlap": len(s_top & set(lt.Feature.head(top))),
                         "LIME local R2": info["local_r2"]})
    d = pd.DataFrame(rows)
    out = d.groupby("Group").agg(**{"Patients": ("Top-5 overlap", "size"),
                                    "Mean top-5 overlap (of 5)": ("Top-5 overlap", "mean"),
                                    "Mean LIME local R2": ("LIME local R2", "mean")})
    out.loc["All"] = [len(d), d["Top-5 overlap"].mean(), d["LIME local R2"].mean()]
    return out.round(2)


def run_experiment(spec, feature_set, attribute, models=MODEL_NAMES, mitigation_models=("Logistic Regression", "XGBoost"),
                   methods=METHODS, repeats_base=5, repeats_mit=5, representation_group=None,
                   out_dir: Path | None = None, log=print, external=None, select_repeats=2) -> dict:
    t0 = time.time()
    X, y, A = spec.get(feature_set, attribute)
    thr, proxy = spec.threshold, spec.proxy_cols.get(attribute, [])
    out_dir = Path(out_dir or RESULTS / slug(spec, feature_set, attribute))
    (out_dir / "tables").mkdir(parents=True, exist_ok=True); (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    R = {"config": {"Dataset": spec.name, "Feature set": feature_set, "Sensitive attribute": attribute,
                    "Patients": len(y), "Features": X.shape[1], "Groups": ", ".join(sorted(A.unique())),
                    "Decision threshold": "training-set CKD prevalence" if thr == "prevalence" else thr,
                    "CV protocol": f"stratified 5-fold, {repeats_base} repeats (baseline), {repeats_mit} repeats (mitigation)"}}
    log(f"[1-3] Data, preprocessing, feature selection: {spec.name} | {feature_set} | {attribute}")
    R["group_summary"] = group_summary(X, y, A)
    R["missing"] = missing_report(X, A)
    R["feature_ranking"] = feature_ranking(X, y, spec.labels)

    log("[4-6] Training models + fairness audit")
    raw, base_preds = {}, {}
    W = spec.weights(attribute)
    for m in models:
        raw[m], pp = cross_validate(X, y, A, m, BASELINE, n_repeats=repeats_base, threshold=thr,
                                    return_predictions=True, weights=W)
        base_preds[m] = pp[0]
        log(f"      {m:20s} acc={raw[m].Accuracy.mean():.3f}  AUC={raw[m].AUC.mean():.3f}  "
            f"EOpp={raw[m]['EOpp diff'].mean():.3f}  EOdds={raw[m]['EOdds diff'].mean():.3f}")
    R["baseline_raw"] = pd.concat([d.assign(Model=m) for m, d in raw.items()], ignore_index=True)
    R["baseline_performance"] = pd.DataFrame({m: summarize(d, PERF) for m, d in raw.items()}).T
    R["baseline_fairness"] = pd.DataFrame({m: summarize(d, FAIR) for m, d in raw.items()}).T
    ref = reference_values(y, A)
    R["baseline_fairness"].loc["Perfect classifier (reference)"] = [f"{ref[c]:.3f}" for c in FAIR]
    R["baseline_group_rates"] = R["baseline_raw"].groupby("Model").mean(numeric_only=True)[
        [c for c in R["baseline_raw"].columns if "[" in c and not c.startswith("W ")]].round(3)
    if W is not None:   # NHANES survey weights -> population-level estimates
        wc = ["W Accuracy", "W Balanced acc.", "W EOpp diff", "W EOdds diff", "W DP diff"] + \
             [c for c in R["baseline_raw"].columns if c.startswith("W TPR [")]
        R["weighted_baseline"] = R["baseline_raw"].groupby("Model", sort=False)[wc].mean().round(3)
    # calibration by group (out-of-fold probabilities, first CV repeat)
    cal_rows, cal_curves = [], []
    for m in [mm for mm in ("Logistic Regression", "XGBoost") if mm in base_preds]:
        t, c = calibration_by_group(y, base_preds[m][1], A, bins=5 if len(y) < 1000 else 10)
        cal_rows.append(t.assign(Model=m)); cal_curves.append(c.assign(Model=m))
    if cal_rows:
        R["calibration"] = pd.concat(cal_rows, ignore_index=True)
        R["calibration_curves"] = pd.concat(cal_curves, ignore_index=True)

    log("[7]   Bias mitigation")
    mraw, mpreds = [], {}
    for m in mitigation_models:
        for meth in methods:
            if meth == UNAWARE and not proxy:
                continue
            if meth == EXPGRAD and m == "KNN":
                continue
            d, pp = cross_validate(X, y, A, m, meth, n_repeats=repeats_mit, threshold=thr, proxy_cols=proxy,
                                   return_predictions=True)
            mraw.append(d.assign(Model=m, Method=meth))
            mpreds[(m, meth)] = pp[0][0]
            log(f"      {m:20s} {SHORT[meth]:15s} acc={d.Accuracy.mean():.3f} bal.acc={d['Balanced acc.'].mean():.3f} "
                f"EOpp={d['EOpp diff'].mean():.3f} EOdds={d['EOdds diff'].mean():.3f}")
    R["mitigation_raw"] = pd.concat(mraw, ignore_index=True)
    cols = ["Accuracy", "Balanced acc.", "Recall", "F1"] + FAIR
    R["mitigation"] = (R["mitigation_raw"].groupby(["Model", "Method"], sort=False)
                       .apply(lambda d: summarize(d, cols), include_groups=False))
    R["mitigation_group_rates"] = R["mitigation_raw"].groupby(["Model", "Method"], sort=False).mean(numeric_only=True)[
        [c for c in R["mitigation_raw"].columns if c.startswith(("TPR [", "FPR ["))]].round(3)

    log("[7b]  Statistical significance (paired bootstrap, 1000 resamples, vs. baseline)")
    sig = []
    for (m, meth), p in mpreds.items():
        if meth == BASELINE or (m, BASELINE) not in mpreds:
            continue
        b = bootstrap_compare(y, A, mpreds[(m, BASELINE)], p, n_boot=1000)
        sig.append({"Model": m, "Method": SHORT[meth],
                    "Δ EOdds gap": round(b["EOdds diff"]["mean"], 3),
                    "95% CI (EOdds)": f"[{b['EOdds diff']['lo']:.3f}, {b['EOdds diff']['hi']:.3f}]",
                    "p (EOdds)": round(b["EOdds diff"]["p"], 3),
                    "Δ Balanced acc.": round(b["Balanced acc."]["mean"], 3),
                    "95% CI (Bal. acc.)": f"[{b['Balanced acc.']['lo']:.3f}, {b['Balanced acc.']['hi']:.3f}]",
                    "p (Bal. acc.)": round(b["Balanced acc."]["p"], 3)})
    R["significance"] = pd.DataFrame(sig)

    log("[7c]  FairCKD-Select (proposed): nested cross-validation")
    sel_raw, picks = nested_select(X, y, A, models=tuple(mm for mm in mitigation_models if mm != "Random Forest"),
                                   threshold=thr, proxy_cols=proxy, outer_repeats=select_repeats,
                                   log=lambda msg: None)
    R["select_raw"], R["select_picks"] = sel_raw, picks
    R["select"] = sel_raw.groupby("Strategy", sort=False).apply(
        lambda d: summarize(d, ["Accuracy", "Balanced acc.", "Recall", "EOpp diff", "EOdds diff", "DP diff"]),
        include_groups=False)
    R["select_picks_summary"] = picks[picks.Strategy == "FairCKD-Select"].groupby(["Model", "Method"]).size() \
        .rename("Times selected").reset_index()
    mm_ = R["mitigation_raw"].groupby(["Model", "Method"], sort=False)[["Balanced acc.", "EOdds diff"]].mean()
    mm_["Pareto-optimal"] = pareto_front(mm_)
    R["pareto"] = mm_.round(3)
    log("      " + " | ".join(f"{k}: bal.acc {v['Balanced acc.']}, EOdds {v['EOdds diff']}" for k, v in
                             R["select"].iterrows()))

    if external is not None:
        log(f"[*]   External validation: train {spec.name} -> test {external.name}")
        Xe, ye, Ae = external.get(feature_set, attribute)
        rows = []
        for m in mitigation_models:
            for meth in [BASELINE, REWEIGH, SMOTE, EXPGRAD, THRESH, REJECT, HYBRID]:
                p, pr = fit_predict(meth, m, X, y, A, Xe, Ae, threshold=thr, proxy_cols=proxy)
                r = fairness_report(ye, p, Ae, pr)
                rows.append({"Model": m, "Method": SHORT[meth], **{k: r[k] for k in
                             ["Accuracy", "Balanced acc.", "Recall", "AUC", "EOpp diff", "EOdds diff"]},
                             **{k: r[k] for k in r if k.startswith("TPR [")}})
        R["external"] = pd.DataFrame(rows).round(3)

    if representation_group:
        log(f"[*]   Representation-bias experiment (under-sampling '{representation_group}')")
        rows = []
        for m in mitigation_models:
            for f in [1.0, 0.5, 0.25, 0.1]:
                for meth in [BASELINE, REWEIGH, SMOTE]:
                    d = cross_validate(X, y, A, m, meth, n_repeats=repeats_mit, threshold=thr,
                                       train_filter=downsample_group(representation_group, f)).mean(numeric_only=True)
                    rows.append({"Model": m, "Kept (%)": int(f * 100), "Method": SHORT[meth],
                                 "Accuracy": d.Accuracy, "EOdds diff": d["EOdds diff"],
                                 **{k: d[k] for k in d.index if k.startswith(("Accuracy [", "TPR ["))}})
        R["representation"] = pd.DataFrame(rows).round(3)

    log("[8]   Explainability (SHAP + LIME)")
    pre = make_preprocessor().fit(X)
    xgb = make_classifier("XGBoost").fit(pre.transform(X), y)
    sv = explain(xgb, pre, X, spec.labels)
    R["shap_by_group"] = importance_by_group(sv, A)
    R["lime_shap_agreement"] = lime_agreement(xgb, pre, X, A, sv, spec)

    log("[9]   Training final fairness-aware model for decision support")
    train_final_model(spec, feature_set, attribute)
    lb = label_bias_table(spec) if attribute == "Race/ethnicity" else None
    if lb is not None:
        R["label_bias"] = lb

    # ------------------------------------------------------------------ save tables
    for k, v in R.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out_dir / "tables" / f"{k}.csv")
    with pd.ExcelWriter(out_dir / "results.xlsx") as w:
        pd.Series(R["config"]).to_frame("Value").to_excel(w, sheet_name="config")
        for k, v in R.items():
            if isinstance(v, pd.DataFrame) and not k.endswith("_raw"):
                v.to_excel(w, sheet_name=k[:31])

    # ------------------------------------------------------------------ figures
    F = {}
    def save(name, fig):
        p = out_dir / "figures" / f"{name}.png"; fig.savefig(p); plots.plt.close(fig); F[name] = p
    groups = sorted(A.unique())
    save("01_group_summary", plots.group_summary(R["group_summary"], f"{spec.name}: {attribute}"))
    save("02_feature_ranking", plots.feature_ranking(R["feature_ranking"]))
    bm = R["baseline_raw"].groupby("Model", sort=False).mean(numeric_only=True)
    save("03_performance", plots.performance(bm[["Accuracy", "Balanced acc.", "F1", "AUC"]], "Baseline model performance"))
    save("04_sensitivity_by_group", plots.group_metric(bm[[f"TPR [{g}]" for g in groups]].rename(columns=lambda c: c[5:-1]),
                                                       "TPR", f"Sensitivity (CKD detection rate) by {attribute.lower()}"))
    save("05_fairness_heatmap", plots.fairness_heatmap(bm[["EOpp diff", "EOdds diff", "PP diff", "Acc. gap", "DP diff"]],
                                                       "Baseline fairness gaps (lower = fairer)"))
    mm = R["mitigation_raw"].groupby(["Model", "Method"], sort=False).mean(numeric_only=True).reset_index()
    mm["Method"] = mm.Method.map(SHORT)
    save("06_mitigation", plots.mitigation(mm))
    save("07_tradeoff", plots.tradeoff(mm))
    if "representation" in R:
        other = [g for g in groups if g != representation_group][0]
        save("08_representation_bias", plots.representation(R["representation"], representation_group, other))
    save("09_shap_summary", plots.shap_summary(sv, "SHAP summary (XGBoost)"))
    save("10_shap_by_group", plots.shap_by_group(R["shap_by_group"], f"Feature reliance by {attribute.lower()} (XGBoost)"))
    if "label_bias" in R:
        save("11_label_bias", plots.label_bias(R["label_bias"]))
    if "calibration_curves" in R:
        save("12_calibration", plots.calibration(R["calibration_curves"], f"Calibration by {attribute.lower()}"))
    save("13_pareto_select", plots.pareto_select(R["pareto"], R["select_raw"]))
    if "external" in R:
        save("14_external_validation", plots.external(R["external"], groups))
    R["figures"] = F
    R["runtime_min"] = round((time.time() - t0) / 60, 1)

    from .report import write_pdf
    write_pdf(R, out_dir / "report.pdf")
    rel = {k: str(Path(p).resolve().relative_to(ROOT)) if Path(p).resolve().is_relative_to(ROOT) else str(p)
           for k, p in F.items()}
    joblib.dump({k: v for k, v in R.items() if k != "figures"} | {"figures": rel},
                out_dir / "results.joblib")
    log(f"Done in {R['runtime_min']} min -> {out_dir}")
    return R
