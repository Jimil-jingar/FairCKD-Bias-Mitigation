"""Stages 5-6: bias detection and fairness evaluation."""
from common import GROUP_COLORS, wrap, group_colors, insight, metric_help, page_header, page_setup, style_table, pd, px, run_store, saved_results, sidebar, st
from fairckd.evaluation import summarize
from fairckd.fairness import FAIR, reference_values

page_setup("Fairness Audit", "⚖️")
page_header("Fairness audit", "Stages 5–6: bias detection and fairness evaluation",
            "Does each model detect CKD equally well in every patient group?")
cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg
raw = st.session_state.get("baseline_raw_" + spec.key + fs + at)
if raw is None:
    saved = saved_results(spec, fs, at)
    raw = saved["baseline_raw"] if saved else None
if raw is None:
    st.warning("Train the models first on the **🤖 Model Training** page.")
    st.stop()

groups = sorted(A.unique())
mean = raw.groupby("Model", sort=False).mean(numeric_only=True)

_avg = {g: mean[f"TPR [{g}]"].mean() for g in groups}
_hi, _lo = max(_avg, key=_avg.get), min(_avg, key=_avg.get)
_gaps = mean[[f"TPR [{g}]" for g in groups]].max(axis=1) - mean[[f"TPR [{g}]" for g in groups]].min(axis=1)
_n_lo = int((mean[[f"TPR [{g}]" for g in groups]].idxmin(axis=1).str[5:-1] == _lo).sum())
insight(f"{100 * (_avg[_hi] - _avg[_lo]):.0f} points",
        f"Across all {len(mean)} models, CKD is detected in <b>{_avg[_hi]:.0%}</b> of real cases in the <b>{_hi}</b> group "
        f"but only <b>{_avg[_lo]:.0%}</b> in the <b>{_lo}</b> group. The <b>{_lo}</b> group comes last in "
        f"<b>{_n_lo} of {len(mean)}</b> models; gaps range from {100 * _gaps.min():.0f} to {100 * _gaps.max():.0f} points.",
        bars=[(g, _avg[g], GROUP_COLORS[i % 6]) for i, g in enumerate(groups)],
        caption="Share of real CKD cases detected, average of all models")
st.markdown("### 1 · Bias detection: does the model work equally well for every group?")
metric = st.radio("Group-wise rate", ["TPR (sensitivity: CKD patients detected)", "FPR (healthy patients wrongly flagged)",
                                      "PPV (precision of a CKD flag)", "Accuracy"], horizontal=True)
mk = metric.split(" ")[0]
d = mean[[f"{mk} [{g}]" for g in groups]].rename(columns=lambda c: c[len(mk) + 2:-1]).reset_index()
long = d.melt("Model", var_name="Group", value_name=mk)
long["Model"] = long.Model.map(wrap)
fig = px.bar(long, x="Model", y=mk, color="Group", barmode="group", height=420,
             title=f"{metric} by {at.lower()}", color_discrete_map=group_colors(groups))
lo = long[mk].min()
fig.update_yaxes(range=[max(0, lo - 0.15) if mk != "FPR" else 0, min(1.02, long[mk].max() + 0.05)])
fig.update_layout(margin=dict(t=50, b=10), legend_title_text="")
st.plotly_chart(fig, width="stretch", theme=None)
st.markdown("### 2 · Fairness metrics")
tab = raw.groupby("Model", sort=False).apply(lambda d: summarize(d, FAIR), include_groups=False)
ref = reference_values(y, A)
tab.loc["Perfect classifier (reference)"] = [f"{ref[c]:.3f}" for c in FAIR]
st.dataframe(style_table(tab, lower_better=["DP diff", "EOpp diff", "EOdds diff", "PP diff", "Acc. gap"], higher_better=["DI ratio"]), width="stretch")
st.caption(f"Reference row = values for a classifier that is always correct. Here even a perfect model has "
           f"DP diff = {ref['DP diff']:.3f} and DI = {ref['DI ratio']:.3f}, because CKD prevalence genuinely differs "
           "between groups, so error-rate metrics (Equal Opportunity, Equalized Odds) are the right fairness criteria "
           "for diagnosis.")
hm = mean[["EOpp diff", "EOdds diff", "PP diff", "Acc. gap", "DP diff"]]
fig = px.imshow(hm, text_auto=".3f", color_continuous_scale=[[0, "#FFFFFF"], [0.35, "#E9B9C3"], [1, "#9F1D35"]], zmin=0, zmax=max(0.3, float(hm.values.max())),
                aspect="auto", title="Fairness gaps (lower = fairer)", height=60 * len(hm) + 120)
fig.update_layout(margin=dict(t=50, b=10))
st.plotly_chart(fig, width="stretch", theme=None)
with st.expander("What do the metrics mean?"):
    st.dataframe(metric_help(), width="stretch", hide_index=True)

st.markdown("### 3 · Detailed group table")
m = st.selectbox("Model", list(mean.index))
rows = []
for g in groups:
    rows.append({"Group": g, **{k: round(mean.loc[m, f"{k} [{g}]"], 3) for k in ["TPR", "FPR", "PPV", "Accuracy", "Selection rate"]}})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
