"""FairCKD - Fairness-aware CKD Diagnosis System (Streamlit entry point).

Run from the project folder:   streamlit run app/Home.py
"""
import joblib
import pandas as pd

from common import RESULTS, page_setup, saved_results, sidebar, st

page_setup("Home")
cfg = sidebar()


# ----------------------------------------------------------------------------- hero: the core finding, from real results
def detection_gap():
    """Sensitivity of a standard XGBoost model for elderly vs younger adults (saved NHANES or UCI run)."""
    for folder, label in [("nhanes__age-group__routine-blood-tests", "5,063 US adults (NHANES 2017-18)"),
                          ("uci__age-group__primary-care-screening", "391 patients (UCI CKD, India)")]:
        p = RESULTS / folder / "results.joblib"
        if p.exists():
            b = joblib.load(p)["baseline_raw"]
            m = b[b.Model == "XGBoost"].mean(numeric_only=True)
            return m["TPR [Elderly (>=60)]"], m["TPR [Non-elderly (<60)]"], label
    return None


gap = detection_gap()
bars = ""
if gap:
    old, young, src = gap
    bars = f"""
  <div class="gap-card">
    <div class="q">Share of real CKD cases a standard XGBoost model detects</div>
    <div class="gap-row"><div class="lbl"><span>Adults 60 and older</span><b>{old:.0%}</b></div>
      <div class="gap-track"><div class="gap-fill" style="width:{100 * old:.0f}%; background:#2BB3A3"></div></div></div>
    <div class="gap-row"><div class="lbl"><span>Adults under 60</span><b>{young:.0%}</b></div>
      <div class="gap-track"><div class="gap-fill" style="width:{100 * young:.0f}%; background:#E0566F"></div></div></div>
    <div class="src">{src}, repeated 5-fold cross-validation</div>
  </div>"""

st.markdown(f"""
<div class="hero">
  <div>
    <h1>Same disease.<br>Unequal detection.</h1>
    <p>Machine-learning models for chronic kidney disease can look accurate overall while missing far more
    cases in some patient groups. FairCKD finds these gaps, measures them with five fairness metrics,
    reduces them with its FairCKD-Select method, and explains every prediction.</p>
    <div class="who">Fairness-aware machine learning for automated chronic kidney disease diagnosis<br>Built by Jimil Jingar,
    guided by Ramizraja Shethwala and Robin Hojiwala. Department of AI &amp; DS, PIET, Parul University</div>
  </div>{bars}
</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- results across experiments
runs = sorted(RESULTS.glob("*/results.joblib"))
if runs:
    from fairckd.report import overview_row
    ov = pd.DataFrame([overview_row(joblib.load(p)) for p in runs])
    def _red(v):
        a_, b_ = (float(x) for x in v.split(" → "))
        return a_ - b_
    done = ov.dropna(subset=["EOdds gap before → after"])
    best = done.loc[done["EOdds gap before → after"].map(_red).idxmax()]
    k = st.columns(4)
    k[0].metric("Experiments run", len(ov))
    k[1].metric("Patients analysed", f"{int(ov.drop_duplicates('Dataset').Patients.sum()):,}")
    k[2].metric("Patient groupings audited", str(ov["Sensitive attribute"].nunique()),
                help="Age group, sex, race/ethnicity and the intersections Age × Sex and Age × Race")
    k[3].metric("Largest fairness gap closed", f"{_red(best['EOdds gap before → after']):.2f}",
                help=f"Equalized Odds gap {best['EOdds gap before → after']} ({best['Sensitive attribute']}, "
                     f"{best['Best mitigation']})")
    st.markdown("## Results of all experiments")
    fs_short = {"Routine blood tests": "routine tests", "Primary-care screening": "screening",
                "Full clinical panel": "full panel"}
    short = lambda m: str(m).replace("Logistic Regression", "Log. Reg.").replace("Random Forest", "Rand. Forest")
    show = pd.DataFrame({
        "Dataset": ov["Dataset"].str.split(" ").str[0] + " · " + ov["Feature set"].map(fs_short).fillna(ov["Feature set"]),
        "Patient groups": ov["Sensitive attribute"],
        "Best model": ov["Best model"].map(short),
        "Bal. acc.": ov["Balanced acc."],
        "Worst gap": ov["Worst EOdds gap (7 models)"],
        "Best mitigation": ov["Best mitigation"].map(short),
        "Gap before → after": ov["EOdds gap before → after"],
        "FairCKD-Select gap": ov["FairCKD-Select gap"],
        "FairCKD-Select bal. acc.": ov["FairCKD-Select bal. acc."],
    })
    def _row(r):
        g = float(r["FairCKD-Select gap"]); w = min(100, g / 0.7 * 100)
        bef, aft = r["Gap before → after"].split(" → ")
        return (f"<tr><td class='l'><b>{r['Dataset']}</b></td><td class='l'>{r['Patient groups']}</td>"
                f"<td class='l'>{r['Best model']}</td><td>{r['Bal. acc.']:.3f}</td><td>{r['Worst gap']:.3f}</td>"
                f"<td class='l'>{r['Best mitigation']}</td><td>{bef} <span class='arr'>→</span> <b>{aft}</b></td>"
                f"<td><div class='gbar'><span style='width:{w:.0f}%'></span></div><b>{g:.3f}</b></td>"
                f"<td>{r['FairCKD-Select bal. acc.']:.3f}</td></tr>")
    head = ("<tr><th class='l'>Dataset</th><th class='l'>Patient groups</th><th class='l'>Best model (baseline)</th>"
            "<th>Balanced acc.</th><th>Worst EOdds gap</th><th class='l'>Best single mitigation</th>"
            "<th>Gap before → after</th><th class='hl'>FairCKD-Select gap</th><th class='hl'>FairCKD-Select bal. acc.</th></tr>")
    st.markdown("<div class='xt'><table>" + head + "".join(_row(r) for _, r in show.iterrows()) + "</table></div>",
                unsafe_allow_html=True)
    st.caption("Best model (baseline) = highest balanced accuracy without any mitigation. Mitigation is compared on "
               "Logistic Regression and XGBoost, so the best single mitigation can use a different model. "
               "Best mitigation is the largest reduction in the Equalized Odds gap that keeps balanced accuracy "
               "within 3 points of the baseline model.")

# ----------------------------------------------------------------------------- framework pipeline (a real sequence)
st.markdown("## The nine-stage framework")
stages = [("Data", "UCI, NHANES or your CSV", 0), ("Preprocess", "impute and scale on training folds", 0),
          ("Features", "MI, correlation, RFE + proxy check", 0), ("Models", "7 classifiers", 0),
          ("Bias detection", "error rates per group", 1), ("Fairness", "5 metrics", 1),
          ("Mitigation", "8 methods + FairCKD-Select", 1), ("Explain", "SHAP, LIME, PDP/ICE", 0), ("Decide", "patient risk + report", 0)]
st.markdown("<div class='pipe'>" + "".join(
    f"<div class='{'fair' if f else ''}'><b>{i}</b>{t}<br><span class='small'>{d}</span></div>"
    for i, (t, d, f) in enumerate(stages, 1)) + "</div>", unsafe_allow_html=True)
st.caption("Stages 5 to 7, in red, are the fairness layer that a standard diagnostic pipeline leaves out.")

# ----------------------------------------------------------------------------- current configuration
if cfg:
    spec, fs, at, X, y, A = cfg
    st.markdown("## Current configuration")
    c = st.columns(4)
    c[0].metric("Patients", f"{len(y):,}"); c[1].metric("Features", X.shape[1])
    c[2].metric("CKD prevalence", f"{100 * y.mean():.1f}%"); c[3].metric("Groups", A.nunique())
    st.caption(spec.description)
    R = saved_results(spec, fs, at)
    if R:
        from fairckd.report import key_findings
        st.markdown("### Key findings for this configuration")
        for f in key_findings(R):
            st.markdown("- " + f.replace("<b>", "**").replace("</b>", "**"))
    st.info("Use the pages in the sidebar in order, from Dataset analysis to Reports. "
            "Change the dataset, feature set or patient group at any time in the sidebar.")

st.caption("Research prototype, not a medical device; predictions must be confirmed by a clinician.")
