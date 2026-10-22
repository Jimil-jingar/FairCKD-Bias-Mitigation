"""Validation and robustness: statistical significance, calibration by group, survey-weighted
population estimates (NHANES) and external validation on NHANES 2015-16."""
from common import (GROUP_COLORS, fig_path, group_colors, insight, page_header, page_setup, pd, px,
                    saved_results, sidebar, st, style_table)

page_setup("Validation")
page_header("Validation", "Robustness checks",
            "Are the fairness gains real, are risk scores equally reliable for every group, and do the "
            "results hold in another population?")
cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg
R = saved_results(spec, fs, at)
if not R:
    st.info("Validation results come from the saved runs (`python main.py --all`). "
            "Choose one of the saved configurations in the sidebar.")
    st.stop()

sig = R.get("significance")
if sig is not None and len(sig):
    ok = sig[(sig["p (EOdds)"] < 0.05) & (sig["Δ EOdds gap"] < 0)]
    best = sig.sort_values("Δ EOdds gap").iloc[0]
    insight(f"{len(ok)} of {len(sig)}",
            "mitigation results reduce the Equalized Odds gap <b>significantly</b> (paired bootstrap over patients, "
            f"1,000 resamples, p &lt; 0.05). The largest reduction is <b>{best['Method']}</b> on <b>{best['Model']}</b>: "
            f"Δ gap {best['Δ EOdds gap']:+.3f}, 95% CI {best['95% CI (EOdds)']}, p = {best['p (EOdds)']:.3f}.")

tabs = ["Statistical significance", "Calibration by group"]
if "weighted_baseline" in R:
    tabs.append("Population-weighted (survey weights)")
if "external" in R:
    tabs.append("External validation (NHANES 2015-16)")
if "fairness_drift" in R:
    tabs.append("Monitoring: drift")
T = st.tabs(tabs)

with T[0]:
    st.markdown("Each mitigated model is compared with the unmitigated baseline on the same patients "
                "(first cross-validation repeat). Δ = method − baseline; a negative Δ gap means fairer.")
    if sig is not None and len(sig):
        import re
        fp = sig.copy()
        ci = fp["95% CI (EOdds)"].map(lambda v: [float(x) for x in re.findall(r"-?\d+\.\d+", str(v))])
        fp["lo"], fp["hi"] = ci.str[0], ci.str[1]
        fp["Pipeline"] = fp["Model"].str.replace("Logistic Regression", "LR") + " · " + fp["Method"]
        fp["Result"] = fp["p (EOdds)"].map(lambda p: "Significant (p < 0.05)" if p < 0.05 else "Not significant")
        fp = fp.iloc[::-1]
        f = px.scatter(fp, x="Δ EOdds gap", y="Pipeline", color="Result", height=90 + 46 * len(fp),
                       error_x=fp["hi"] - fp["Δ EOdds gap"], error_x_minus=fp["Δ EOdds gap"] - fp["lo"],
                       color_discrete_map={"Significant (p < 0.05)": "#0F766E", "Not significant": "#94A3B8"},
                       title="Change in Equalized Odds gap vs. baseline, with 95% bootstrap CI (left of 0 = fairer)")
        f.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        f.add_vline(x=0, line=dict(color="#9F1D35", dash="dot"))
        f.update_layout(legend_title_text="", yaxis_title=None, margin=dict(t=60, b=10),
                        legend=dict(orientation="h", y=-0.12 if len(fp) > 6 else -0.2, x=0))
        st.plotly_chart(f, width="stretch", theme=None)
        st.dataframe(style_table(sig, lower_better=["p (EOdds)"]), width="stretch", hide_index=True)

with T[1]:
    cal = R.get("calibration"); cur = R.get("calibration_curves")
    if cal is not None:
        st.markdown("A well-calibrated model's predicted risk matches the observed CKD rate in **every** group. "
                    "Calibration-in-the-large is mean predicted minus observed risk (percentage points); ECE is the "
                    "expected calibration error.")
        st.dataframe(style_table(cal, lower_better=["ECE", "Brier"]), width="stretch", hide_index=True)
        m = st.selectbox("Model", list(cur.Model.unique()))
        d = cur[(cur.Model == m) & (cur["n"] >= 5)]
        f = px.line(d, x="Mean predicted", y="Observed CKD rate", color="Group", markers=True, height=420,
                    color_discrete_map=group_colors(d.Group.unique()), title=f"Reliability curve by {at.lower()}: {m}")
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
        f.update_xaxes(range=[0, 1]); f.update_yaxes(range=[0, 1]); f.update_layout(legend_title_text="")
        st.plotly_chart(f, width="stretch", theme=None)

if "weighted_baseline" in R:
    with T[tabs.index("Population-weighted (survey weights)")]:
        st.markdown("NHANES is a complex survey. Applying the examination weights (WTMEC2YR) turns sample "
                    "results into estimates for the **US adult population**.")
        st.dataframe(style_table(R["weighted_baseline"], lower_better=["W EOpp diff", "W EOdds diff", "W DP diff"],
                                 higher_better=["W Balanced acc."]), width="stretch")

if "external" in R:
    with T[tabs.index("External validation (NHANES 2015-16)")]:
        ex = R["external"]
        b0 = ex[ex.Method == "Baseline"]
        st.markdown(f"Models were trained on all NHANES 2017-18 adults and tested **once** on 5,309 adults from "
                    f"NHANES 2015-16, a separate survey cycle. Baseline balanced accuracy on the new cohort: "
                    f"**{b0['Balanced acc.'].min():.3f}–{b0['Balanced acc.'].max():.3f}**.")
        st.dataframe(style_table(ex, lower_better=["EOpp diff", "EOdds diff"], higher_better=["Balanced acc.", "AUC"]),
                     width="stretch", hide_index=True)
        if "14_external_validation" in R["figures"]:
            st.image(fig_path(R["figures"]["14_external_validation"]))

if "fairness_drift" in R:
    with T[tabs.index("Monitoring: drift")]:
        fd, dd, pv = R["fairness_drift"], R["data_drift"], R["prevalence_drift"]
        held = (fd["Fairness held?"] == "Yes").sum()
        shifted = (dd.Shift != "Stable").sum()
        st.markdown(
            "A model that was fair at validation can become unfair after deployment when the patient mix, the "
            "measurements or the disease prevalence change. Here the 2015-16 NHANES cohort plays the role of "
            f"new patients. **{held} of {len(fd)}** pipelines keep their Equalized Odds gap within 0.05 of the "
            f"development estimate, and **{shifted}** feature(s) shift at least moderately inside some group "
            "(PSI > 0.1). This is why FairCKD recommends re-running the fairness audit on new data at regular intervals.")
        d = fd.copy(); d["Pipeline"] = d.Model.str.replace("Logistic Regression", "LR") + " · " + d.Method
        long = d.melt("Pipeline", ["EOdds diff (internal)", "EOdds diff (external)"], var_name="Cohort",
                      value_name="EOdds gap")
        long["Cohort"] = long.Cohort.map({"EOdds diff (internal)": "Development (CV, 2017-18)",
                                          "EOdds diff (external)": "External (2015-16)"})
        f = px.bar(long, x="EOdds gap", y="Pipeline", color="Cohort", barmode="group", orientation="h",
                   height=46 * len(d) + 140, title="Fairness drift: Equalized Odds gap per pipeline (lower is fairer)",
                   color_discrete_map={"Development (CV, 2017-18)": "#0F766E", "External (2015-16)": "#9F1D35"})
        f.update_yaxes(title=None, autorange="reversed")
        f.update_layout(legend_title_text="", margin=dict(t=60, b=10), legend=dict(orientation="h", y=-0.08, x=0))
        st.plotly_chart(f, width="stretch", theme=None)
        st.dataframe(style_table(fd, lower_better=["EOdds diff (internal)", "EOdds diff (external)", "Δ EOdds gap"]),
                     width="stretch", hide_index=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**CKD prevalence per group in each cohort**")
            st.dataframe(pv, width="stretch", hide_index=True)
            if "sensitivity_drift" in R and len(R["sensitivity_drift"]):
                st.markdown("**Sensitivity per group: development vs external (unmitigated models)**")
                st.dataframe(R["sensitivity_drift"], width="stretch", hide_index=True)
        with c2:
            top = dd.head(10)[::-1]
            f = px.bar(top, x="Max group PSI", y="Feature", orientation="h", color="Shift", height=380,
                       color_discrete_map={"Significant": "#9F1D35", "Moderate": "#C27C0E", "Stable": "#94A3B8"},
                       title="Data drift: largest within-group PSI per feature")
            f.add_vline(x=0.1, line=dict(dash="dot", color="#C27C0E")); f.add_vline(x=0.2, line=dict(dash="dot", color="#9F1D35"))
            f.update_yaxes(title=None); f.update_layout(legend_title_text="", margin=dict(t=50, b=10))
            st.plotly_chart(f, width="stretch", theme=None)
            st.caption("Population Stability Index: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.")
        with st.expander("Full data-drift table (PSI overall and per group)"):
            st.dataframe(dd, width="stretch", hide_index=True)
