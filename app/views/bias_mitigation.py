"""Stage 7: bias reduction - compare pre-, in- and post-processing mitigation."""
from common import METHOD_COLORS, wrap, fig_path, insight, page_header, page_setup, style_table, pd, px, run_store, saved_results, sidebar, st
from fairckd.evaluation import cross_validate, summarize
from fairckd.fairness import FAIR
from fairckd.mitigation import EXPGRAD, METHODS, SHORT, UNAWARE
from fairckd.models import MODEL_NAMES

page_setup("Bias Mitigation", "🛠️")
page_header("Bias mitigation", "Stage 7: bias reduction",
            "Compare pre-, in- and post-processing methods and see what each one costs in accuracy.")
cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg
store = run_store(spec, fs, at)
saved = saved_results(spec, fs, at)
proxy = spec.proxy_cols.get(at, [])

with st.expander("About the methods", expanded=False):
    st.markdown("""
| Method | Type | Idea |
|---|---|---|
| **Unawareness** | naive | remove the sensitive attribute from the model inputs |
| **Reweighing** (Kamiran & Calders 2012) | pre-processing | weight training samples so that group and label become independent |
| **Exponentiated Gradient** (Agarwal et al. 2018) | in-processing | train under an Equalized-Odds constraint |
| **Threshold Optimizer** (Hardt et al. 2016) | post-processing | a separate decision threshold per group, learned on out-of-fold scores |
""")

src = "Saved thesis results" if saved else "Run live"
if saved:
    src = st.radio("Results source", ["Saved thesis results", "Run live"], horizontal=True)
if src == "Run live":
    c = st.columns([2, 3, 1])
    models = c[0].multiselect("Models", MODEL_NAMES, default=["Logistic Regression", "XGBoost"])
    meths = c[1].multiselect("Methods", METHODS, default=METHODS)
    reps = c[2].number_input("CV repeats", 1, 10, 1 if len(y) > 2000 else 3)
    if st.button("▶ Run mitigation comparison", type="primary"):
        out, todo = [], [(m, k) for m in models for k in meths
                         if not (k == UNAWARE and not proxy) and not (k == EXPGRAD and m == "KNN")]
        pb = st.progress(0.0)
        for i, (m, k) in enumerate(todo):
            pb.progress(i / len(todo), f"{m} · {SHORT[k]}…")
            out.append(cross_validate(X, y, A, m, k, n_repeats=int(reps), threshold=spec.threshold,
                                      proxy_cols=proxy).assign(Model=m, Method=k))
        pb.progress(1.0, "Done"); store["mitigation"] = pd.concat(out, ignore_index=True)
    raw = store.get("mitigation")
    if raw is None:
        st.info("Select models and methods, then press **Run mitigation comparison**. "
                "Exponentiated Gradient is the slowest method.")
        st.stop()
else:
    raw = saved["mitigation_raw"]

cols = ["Accuracy", "Balanced acc.", "Recall"] + FAIR
_mm = raw.groupby(["Model", "Method"], sort=False).mean(numeric_only=True)
_best = None
for _mod in _mm.index.get_level_values(0).unique():
    _d = _mm.loc[_mod]; _b = _d.loc["Baseline"] if "Baseline" in _d.index else None
    if _b is None:
        continue
    _ok = _d.drop(index="Baseline"); _ok = _ok[_ok["Balanced acc."] >= _b["Balanced acc."] - 0.03]
    if len(_ok):
        _k = _ok["EOdds diff"].idxmin(); _red = _b["EOdds diff"] - _d.loc[_k, "EOdds diff"]
        if _best is None or _red > _best[0]:
            _best = (_red, _mod, _k, _b, _d.loc[_k])
if _best and _best[0] > 0:
    _red, _mod, _k, _b, _a = _best
    _pct = _red / _b["EOdds diff"] if _b["EOdds diff"] > 0 else 0
    insight(f"{_b['EOdds diff']:.3f} → {_a['EOdds diff']:.3f}",
            f"<b>{SHORT[_k]}</b> on <b>{_mod}</b> makes the fairness gap <b>{_pct:.0%} smaller</b> "
            f"(Equalized Odds difference), at a cost of {100 * (_b['Balanced acc.'] - _a['Balanced acc.']):.1f} points of "
            "balanced accuracy. This is the largest reduction that keeps the model clinically useful (≤ 3 points lost).",
            bars=[(f"Before: {_mod} baseline", _b["EOdds diff"], "maroon"), (f"After: {SHORT[_k]}", _a["EOdds diff"], "teal")],
            caption=f"Equalized Odds gap, lower is fairer · balanced accuracy {_b['Balanced acc.']:.3f} → {_a['Balanced acc.']:.3f}",
            fmt="{:.3f}")
else:
    insight("No safe fix", "No mitigation method reduced the fairness gap without losing more than 3 points of balanced "
            "accuracy for this configuration. The trade-off chart below shows the options.")
st.markdown("### Comparison table (mean ± SD over CV repeats)")
st.dataframe(style_table(raw.assign(Method=raw.Method.map(SHORT)).groupby(["Model", "Method"], sort=False).apply(lambda d: summarize(d, cols), include_groups=False), lower_better=["DP diff", "EOpp diff", "EOdds diff", "PP diff", "Acc. gap"], higher_better=["Accuracy", "Balanced acc.", "Recall", "DI ratio"]),
             width="stretch")

mm = raw.groupby(["Model", "Method"], sort=False).mean(numeric_only=True).reset_index()
mm["Method"] = mm.Method.map(SHORT)
c1, c2 = st.columns(2)
for c, met in [(c1, "EOpp diff"), (c2, "EOdds diff")]:
    f = px.bar(mm.assign(Model=mm.Model.map(wrap)), x="Model", y=met, color="Method", barmode="group",
               color_discrete_map=METHOD_COLORS, title=f"{met} (lower = fairer)", height=420)
    f.update_layout(margin=dict(t=50, b=10, r=10), legend_title_text="", xaxis_title=None,
                    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=11)))
    c.plotly_chart(f, width="stretch", theme=None)

acc = "Balanced acc."
f = px.scatter(mm, x="EOdds diff", y=acc, color="Method", symbol="Model", color_discrete_map=METHOD_COLORS,
               title="Fairness–accuracy trade-off (left = fairer, up = more accurate)", height=430)
f.update_traces(marker=dict(size=14, line=dict(width=1, color="black")))
f.update_layout(margin=dict(t=50, b=10))
st.plotly_chart(f, width="stretch", theme=None)

st.markdown("### Recommendation")
for m in mm.Model.unique():
    d = mm[mm.Model == m].set_index("Method")
    base = d.loc["Baseline"]
    ok = d.drop(index="Baseline")
    ok = ok[ok[acc] >= base[acc] - 0.03]
    if ok.empty:
        st.markdown(f"- **{m}**: no method reduced the gap without losing more than 3 points of balanced accuracy.")
        continue
    b = ok["EOdds diff"].idxmin()
    st.markdown(f"- **{m}** → **{b}**: Equalized Odds gap {base['EOdds diff']:.3f} → {d.loc[b, 'EOdds diff']:.3f}, "
                f"balanced accuracy {base[acc]:.3f} → {d.loc[b, acc]:.3f}")

with st.expander("Group-wise sensitivity / false-positive rate after mitigation"):
    g = raw.groupby(["Model", "Method"], sort=False).mean(numeric_only=True)
    st.dataframe(g[[c for c in g.columns if c.startswith(("TPR [", "FPR ["))]].round(3), width="stretch")

if saved and "select" in saved:
    st.markdown("### FairCKD-Select: the proposed method")
    sr = saved["select_raw"].groupby("Strategy").mean(numeric_only=True)
    if {"FairCKD-Select", "Accuracy-driven selection"} <= set(sr.index):
        f_, a_ = sr.loc["FairCKD-Select"], sr.loc["Accuracy-driven selection"]
        insight(f"{a_['EOdds diff']:.3f} → {f_['EOdds diff']:.3f}",
                "<b>FairCKD-Select</b> picks, inside every training fold, the classifier and mitigation method with the "
                "smallest fairness gap whose balanced accuracy stays within 3 points of the best unmitigated model. "
                f"Evaluated with nested cross-validation it changes the Equalized Odds gap from {a_['EOdds diff']:.3f} "
                f"(conventional accuracy-driven choice) to <b>{f_['EOdds diff']:.3f}</b>, with balanced accuracy "
                f"{a_['Balanced acc.']:.3f} → {f_['Balanced acc.']:.3f}.",
                bars=[("Accuracy-driven selection", a_["EOdds diff"], "maroon"), ("FairCKD-Select", f_["EOdds diff"], "teal")],
                caption="Equalized Odds gap on held-out folds (nested CV), lower is fairer", fmt="{:.3f}")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("**Results (nested cross-validation, mean ± SD)**")
        st.dataframe(saved["select"], width="stretch")
        st.markdown("**Pipelines chosen across outer folds**")
        st.dataframe(saved["select_picks_summary"], width="stretch", hide_index=True)
    with c2:
        if "13_pareto_select" in saved["figures"]:
            st.image(fig_path(saved["figures"]["13_pareto_select"]))
    with st.expander("How FairCKD-Select works"):
        st.markdown("""
1. For every candidate pipeline (classifier × mitigation method, including the new **RW + TO hybrid**), inner
   cross-validation on the training data estimates balanced accuracy and the Equalized Odds gap.
2. The accuracy reference is the best unmitigated model; candidates more than 3 points below it are rejected
   (clinical-utility tolerance δ = 0.03).
3. Among the remaining candidates, the one with the smallest fairness gap is chosen and refitted.
4. This whole procedure runs inside an outer cross-validation, so the reported result is unbiased.
""")

if saved and "representation" in saved:
    st.markdown("### Representation-bias experiment (saved thesis run)")
    st.caption("One group is under-sampled in the TRAINING data only; test data are unchanged.")
    st.image(fig_path(saved["figures"]["08_representation_bias"]))
    st.dataframe(saved["representation"], width="stretch", hide_index=True)
