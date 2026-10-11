"""Automatic findings + PDF report (ReportLab)."""
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .mitigation import BASELINE, SHORT


def key_findings(R) -> list[str]:
    out = []
    b = R["baseline_raw"].groupby("Model", sort=False).mean(numeric_only=True)
    best = b["Balanced acc."].idxmax()
    out.append(f"Best baseline model: <b>{best}</b> (accuracy {b.loc[best, 'Accuracy']:.3f}, "
               f"balanced accuracy {b.loc[best, 'Balanced acc.']:.3f}, AUC {b.loc[best, 'AUC']:.3f}).")
    tpr = b[[c for c in b.columns if c.startswith("TPR [")]]
    worst = tpr.idxmin(axis=1).str[5:-1]
    g, n = worst.value_counts().idxmax(), worst.value_counts().max()
    out.append(f"Bias detection: in <b>{n} of {len(b)}</b> models, CKD is detected least often in the "
               f"<b>{g}</b> group (Equal Opportunity gap {b['EOpp diff'].min():.3f}–{b['EOpp diff'].max():.3f}).")
    ref = R["baseline_fairness"].loc["Perfect classifier (reference)"]
    out.append(f"Even a perfect classifier has DP diff = {ref['DP diff']} and DI = {ref['DI ratio']} here, because CKD "
               "prevalence differs between groups; error-rate metrics (Equal Opportunity, Equalized Odds) are the "
               "clinically meaningful fairness criteria.")
    m = R["mitigation_raw"].groupby(["Model", "Method"], sort=False).mean(numeric_only=True)
    for model in m.index.get_level_values(0).unique():
        d = m.loc[model]
        base = d.loc[BASELINE]
        ok = d.drop(index=BASELINE)
        ok = ok[ok["Balanced acc."] >= base["Balanced acc."] - 0.03]      # must stay clinically useful
        if ok.empty:
            out.append(f"{model}: no mitigation reduced the gap without losing more than 3 points of balanced accuracy.")
            continue
        bestm = ok["EOdds diff"].idxmin()
        out.append(f"{model}: best mitigation (balanced accuracy loss ≤ 3 points) = <b>{SHORT[bestm]}</b>, Equalized Odds gap "
                   f"{base['EOdds diff']:.3f} → {d.loc[bestm, 'EOdds diff']:.3f}, balanced accuracy "
                   f"{base['Balanced acc.']:.3f} → {d.loc[bestm, 'Balanced acc.']:.3f}.")
    if "select" in R and "select_raw" in R:
        sr = R["select_raw"].groupby("Strategy").mean(numeric_only=True)
        if {"FairCKD-Select", "Accuracy-driven selection"} <= set(sr.index):
            f_, a_ = sr.loc["FairCKD-Select"], sr.loc["Accuracy-driven selection"]
            out.append(f"<b>FairCKD-Select</b> (proposed, nested CV): Equalized Odds gap {a_['EOdds diff']:.3f} → "
                       f"<b>{f_['EOdds diff']:.3f}</b> compared with accuracy-driven model selection, balanced accuracy "
                       f"{a_['Balanced acc.']:.3f} → {f_['Balanced acc.']:.3f}.")
    if "significance" in R and len(R["significance"]):
        sg = R["significance"]
        ok = sg[(sg["p (EOdds)"] < 0.05) & (sg["Δ EOdds gap"] < 0)]
        out.append(f"Statistical significance: {len(ok)} of {len(sg)} mitigation results reduce the Equalized Odds gap "
                   "significantly (paired bootstrap, p < 0.05).")
    if "external" in R:
        ex = R["external"]; b0 = ex[ex.Method == "Baseline"]
        out.append(f"External validation (NHANES 2015–16): baseline balanced accuracy {b0['Balanced acc.'].min():.3f}–"
                   f"{b0['Balanced acc.'].max():.3f}, Equalized Odds gap {b0['EOdds diff'].min():.3f}–{b0['EOdds diff'].max():.3f}; "
                   f"best mitigated gap {ex['EOdds diff'].min():.3f}.")
    if "label_bias" in R:
        lb = R["label_bias"]
        bl = lb.loc["Non-Hispanic Black"] if "Non-Hispanic Black" in lb.index else None
        if bl is not None:
            out.append(f"Label bias: the race-based 2009 eGFR equation changes CKD prevalence for Non-Hispanic Black adults "
                       f"from {bl.iloc[1]:.1f}% to {bl.iloc[2]:.1f}% ({int(bl.iloc[4])} patients relabelled), i.e. the "
                       "ground-truth label itself can carry bias.")
    return out


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() and abs(v) >= 1 else f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) >= 1 else f"{v:.3f}"
    return str(v)


def _table(df: pd.DataFrame, index=True, font=7, col0=None, widths=None):
    d = df.reset_index() if index else df.copy()
    d.columns = ["" if str(c) in ("index", "level_0") else str(c) for c in d.columns]
    data = [list(d.columns)] + [[_fmt(v) for v in r] for r in d.values]
    t = Table(data, repeatRows=1, colWidths=widths)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F8")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def write_pdf(R, path: Path):
    ss = getSampleStyleSheet()
    H1, H2, P = ss["Title"], ss["Heading2"], ParagraphStyle("p", parent=ss["Normal"], fontSize=9.5, leading=13)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="FairCKD Fairness Audit Report")
    W = A4[0] - 3.2 * cm
    s = [Paragraph("FairCKD: Fairness Audit Report", H1),
         Paragraph("Investigating Bias and Fairness in Machine Learning Models for Automated CKD Diagnosis<br/>"
                   "M.Tech Dissertation Phase-II · Jimil Jingar · PIET, Parul University", P),
         Spacer(1, 10), _table(pd.Series(R["config"]).to_frame("Value"), font=8), Spacer(1, 10),
         Paragraph("Key findings", H2)]
    s += [Paragraph("• " + f, P) for f in key_findings(R)]

    def fig(name, w=W):
        p = R["figures"].get(name)
        if p:
            from PIL import Image as PI
            iw, ih = PI.open(p).size
            maxh = A4[1] - 5.5 * cm                      # never taller than the page frame
            if w * ih / iw > maxh:
                w = maxh * iw / ih
            # embed a 200-dpi JPEG copy: sharp in print, keeps the PDF small
            import io
            px_w = int(w / 72 * 200)
            im = PI.open(p).convert("RGB")
            if iw > px_w:
                im = im.resize((px_w, int(ih * px_w / iw)), PI.LANCZOS)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); buf.seek(0)
            s.append(Image(buf, width=w, height=w * ih / iw)); s.append(Spacer(1, 6))

    s += [Paragraph("1. Dataset and groups", H2), _table(R["group_summary"], index=False, font=8)]
    fig("01_group_summary")
    s += [PageBreak(), Paragraph("2. Baseline model performance", H2), _table(R["baseline_performance"])]
    fig("03_performance")
    s += [Paragraph("3. Bias detection and fairness evaluation", H2), _table(R["baseline_fairness"])]
    fig("04_sensitivity_by_group"); fig("05_fairness_heatmap", W * 0.8)
    mt = R["mitigation"].rename(index=SHORT, level=1) if isinstance(R["mitigation"].index, pd.MultiIndex) else R["mitigation"]
    perf_c = [c for c in ["Accuracy", "Balanced acc.", "Recall", "F1"] if c in mt.columns]
    fair_c = [c for c in ["EOpp diff", "EOdds diff", "PP diff", "DP diff", "DI ratio", "Acc. gap"] if c in mt.columns]
    s += [PageBreak(), Paragraph("4. Bias mitigation", H2),
          Paragraph("<b>a) Performance</b> (mean ± SD over CV repeats)", P), _table(mt[perf_c], font=7), Spacer(1, 6),
          Paragraph("<b>b) Fairness gaps</b> (lower = fairer; DI ratio: 1 = fair)", P), _table(mt[fair_c], font=6.5), Spacer(1, 6)]
    fig("06_mitigation"); fig("07_tradeoff", W * 0.75)
    if "significance" in R and len(R["significance"]):
        s += [Paragraph("<b>c) Statistical significance</b> (paired bootstrap vs. baseline, 1000 resamples; Δ = method − baseline)", P),
              _table(R["significance"], index=False, font=6.5), Spacer(1, 6)]
    if "select" in R:
        s += [PageBreak(), Paragraph("4b. FairCKD-Select (proposed method)", H2),
              Paragraph("FairCKD-Select chooses, inside each training fold, the classifier and mitigation method with the "
                        "smallest Equalized Odds gap among those whose balanced accuracy is within 3 points of the best "
                        "unmitigated model (nested cross-validation). It is compared with conventional accuracy-driven selection.", P),
              _table(R["select"], font=7), Spacer(1, 4),
              Paragraph("<b>Pipelines chosen by FairCKD-Select across outer folds</b>", P),
              _table(R["select_picks_summary"], index=False, font=7)]
        fig("13_pareto_select", W * 0.85)
    if "calibration" in R:
        s += [Paragraph("4c. Calibration by group", H2), _table(R["calibration"].round(3), index=False, font=6.5)]
        fig("12_calibration", W * 0.85)
    if "weighted_baseline" in R:
        s += [Paragraph("4d. Population-weighted estimates (NHANES survey weights)", H2),
              _table(R["weighted_baseline"], font=6.5)]
    if "external" in R:
        s += [PageBreak(), Paragraph("4e. External validation (trained on NHANES 2017–18, tested on NHANES 2015–16)", H2),
              _table(R["external"].drop(columns=[c for c in R["external"].columns if c.startswith("TPR [")]), index=False, font=7)]
        fig("14_external_validation")
    if "representation" in R:
        s += [Paragraph("5. Representation-bias experiment", H2)]; fig("08_representation_bias")
    s += [PageBreak(), Paragraph("6. Explainability (SHAP)", H2)]
    fig("10_shap_by_group", W * 0.75); fig("09_shap_summary", W * 0.7)
    if "label_bias" in R:
        s += [Paragraph("7. Label bias (eGFR equation)", H2), _table(R["label_bias"].rename(columns={"CKD % (2021 race-free eGFR)": "CKD % 2021\n(race-free)", "CKD % (2009 race-based eGFR)": "CKD % 2009\n(race-based)", "Change (percentage points)": "Change\n(pp)", "Patients whose label changes": "Labels\nchanged"}), font=8)]; fig("11_label_bias", W * 0.8)
    s += [Spacer(1, 8), Paragraph("Protocol: repeated stratified 5-fold cross-validation (stratified on label × group); "
                                  "preprocessing and mitigation fitted on training folds only; mean ± SD over repeats. "
                                  "This system is a research prototype and not a medical device.", P)]
    doc.build(s)


def patient_report_pdf(bundle, values: dict, prob: float, positive: bool, waterfall_png: str | None = None,
                       patient_id: str = "", group: str | None = None) -> bytes:
    """One-page clinical decision-support report for a single patient (returns PDF bytes)."""
    import datetime as dt
    import io
    ss = getSampleStyleSheet()
    P = ParagraphStyle("p", parent=ss["Normal"], fontSize=9.5, leading=13)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm, title="FairCKD Patient Report")
    W = A4[0] - 3.2 * cm
    verdict_col = colors.HexColor("#C0504D") if positive else colors.HexColor("#3A7D44")
    verdict = "CKD LIKELY - confirmatory testing recommended" if positive else "CKD UNLIKELY at this time"
    s = [Paragraph("FairCKD: Patient Decision-Support Report", ss["Title"]),
         Paragraph(f"Generated {dt.datetime.now():%d %b %Y, %H:%M} · Patient ID: {patient_id or '-'}", P), Spacer(1, 8)]
    head = Table([["Probability of CKD", f"{100 * prob:.1f} %"], ["Decision threshold", f"{100 * bundle['threshold']:.1f} %"],
                  ["Result", verdict]], colWidths=[5 * cm, W - 5 * cm])
    head.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                              ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F8")),
                              ("TEXTCOLOR", (1, 2), (1, 2), verdict_col), ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold")]))
    s += [head, Spacer(1, 8)]
    info = [["Model", bundle["model_name"]], ["Training data", f"{bundle['dataset']} ({bundle['n_train']:,} patients)"],
            ["Feature set", bundle["feature_set"]], ["Fairness attribute", bundle["attribute"] + (f"  (patient group: {group})" if group else "")]]
    t = Table(info, colWidths=[5 * cm, W - 5 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F8"))]))
    s += [t, Spacer(1, 8), Paragraph("Patient values", ss["Heading3"])]
    rows, labels = [], bundle["labels"]
    for f in bundle["features"]:
        v = values.get(f, np.nan)
        txt = "not available (imputed)" if v is None or (isinstance(v, float) and np.isnan(v)) else \
            (("Yes" if v >= 0.5 else "No") if f in bundle["binary"] else f"{v:g}")
        rows.append([labels.get(f, f), txt])
    half = (len(rows) + 1) // 2
    grid = [rows[i] + (rows[i + half] if i + half < len(rows) else ["", ""]) for i in range(half)]
    vt = Table(grid, colWidths=[W * 0.32, W * 0.18, W * 0.32, W * 0.18])
    vt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    s.append(vt)
    if waterfall_png:
        from PIL import Image as PI
        iw, ih = PI.open(waterfall_png).size
        w = W * 0.8
        s += [Spacer(1, 6), Paragraph("Why this prediction? (SHAP explanation)", ss["Heading3"]),
              Image(waterfall_png, width=w, height=w * ih / iw),
              Paragraph("Red bars push the prediction towards CKD, blue bars away from CKD.", P)]
    s += [Spacer(1, 8), Paragraph("<i>Research prototype (M.Tech dissertation, PIET, Parul University). Not a medical "
                                  "device; the result must be confirmed by a clinician with eGFR and urine ACR testing.</i>", P)]
    doc.build(s)
    return buf.getvalue()


def overview_row(R) -> dict:
    """One summary row per saved experiment (used by the dashboard)."""
    c = R["config"]
    b = R["baseline_raw"].groupby("Model", sort=False).mean(numeric_only=True)
    best = b["Balanced acc."].idxmax()
    m = R["mitigation_raw"].groupby(["Model", "Method"], sort=False).mean(numeric_only=True)
    row = {"Dataset": c["Dataset"].split(" (")[0], "Sensitive attribute": c["Sensitive attribute"],
           "Feature set": c["Feature set"].split(" (")[0], "Patients": c["Patients"],
           "Best model": best, "Balanced acc.": round(b.loc[best, "Balanced acc."], 3),
           "AUC": round(b.loc[best, "AUC"], 3), "Worst EOdds gap (7 models)": round(b["EOdds diff"].max(), 3)}
    best_red = None
    for model in m.index.get_level_values(0).unique():
        d = m.loc[model]; base = d.loc[BASELINE]
        ok = d.drop(index=BASELINE); ok = ok[ok["Balanced acc."] >= base["Balanced acc."] - 0.03]
        if ok.empty:
            continue
        k = ok["EOdds diff"].idxmin()
        red = base["EOdds diff"] - d.loc[k, "EOdds diff"]
        if best_red is None or red > best_red[0]:
            best_red = (red, model, k, base["EOdds diff"], d.loc[k, "EOdds diff"])
    if "select_raw" in R:
        sr = R["select_raw"].groupby("Strategy").mean(numeric_only=True)
        if "FairCKD-Select" in sr.index:
            row["FairCKD-Select gap"] = round(sr.loc["FairCKD-Select", "EOdds diff"], 3)
            row["FairCKD-Select bal. acc."] = round(sr.loc["FairCKD-Select", "Balanced acc."], 3)
    if best_red:
        row["Best mitigation"] = f"{best_red[1]} + {SHORT[best_red[2]]}"
        row["EOdds gap before → after"] = f"{best_red[3]:.3f} → {best_red[4]:.3f}"
    return row
