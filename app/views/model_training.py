"""Stage 4: model development and evaluation."""
from common import SEQ, wrap, insight, page_header, page_setup, style_table, pd, px, run_store, saved_results, sidebar, st
from fairckd.evaluation import cross_validate, summarize
from fairckd.fairness import PERF
from fairckd.mitigation import BASELINE
from fairckd.models import MODEL_NAMES

page_setup("Model Training", "🤖")
page_header("Model training", "Stage 4: model development",
            "Seven classifiers, each scored with repeated stratified 5-fold cross-validation.")
cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg
store = run_store(spec, fs, at)
saved = saved_results(spec, fs, at)

src = "Saved results" if saved else "Run live"
if saved:
    src = st.radio("Results source", ["Saved results", "Run live"], horizontal=True,
                   help="Saved results come from `python main.py --all` (more CV repeats).")

if src == "Run live":
    c = st.columns([3, 1])
    models = c[0].multiselect("Models", MODEL_NAMES, default=MODEL_NAMES)
    reps = c[1].number_input("CV repeats", 1, 10, 1 if len(y) > 2000 else 3)
    if st.button("▶ Train & evaluate", type="primary"):
        res, bar_ = {}, st.progress(0.0, "Training…")
        for i, m in enumerate(models):
            bar_.progress(i / len(models), f"Training {m}…")
            res[m] = cross_validate(X, y, A, m, BASELINE, n_repeats=int(reps), threshold=spec.threshold)
        bar_.progress(1.0, "Done"); store["baseline"] = res
    res = store.get("baseline")
    if not res:
        st.info("Choose models and press **Train & evaluate**.")
        st.stop()
    raw = pd.concat([d.assign(Model=m) for m, d in res.items()], ignore_index=True)
else:
    raw = saved["baseline_raw"]
st.session_state["baseline_raw_" + spec.key + fs + at] = raw

thr = "training-set CKD prevalence (imbalanced data)" if spec.threshold == "prevalence" else spec.threshold
st.markdown(f"Decision threshold: **{thr}**")
tab = raw.groupby("Model", sort=False).apply(lambda d: summarize(d, PERF), include_groups=False)
_m = raw.groupby("Model", sort=False)[["Accuracy", "Balanced acc."]].mean().sort_values("Balanced acc.", ascending=False)
insight(f"{_m['Balanced acc.'].iloc[0]:.1%}",
        f"<b>{_m.index[0]}</b> has the best balanced accuracy of the {len(_m)} models. Overall scores like this can "
        "still hide a model that works far better for one patient group than another: the Fairness audit checks that.",
        bars=[(m, v, "teal" if i == 0 else "slate") for i, (m, v) in enumerate(_m["Balanced acc."].items())],
        caption="Balanced accuracy, mean over cross-validation repeats", fmt="{:.1%}")
st.dataframe(style_table(tab, higher_better=PERF), width="stretch")
mean = raw.groupby("Model", sort=False)[["Accuracy", "Balanced acc.", "F1", "AUC"]].mean().reset_index()
long = mean.melt("Model", var_name="Metric", value_name="Score")
fig = px.bar(long.assign(Model=long.Model.map(wrap)), x="Model", y="Score", color="Metric", barmode="group", height=400,
             title="Model performance (mean over CV repeats)",
             color_discrete_sequence=SEQ)
fig.update_yaxes(range=[max(0, long.Score.min() - 0.1), 1.01]); fig.update_layout(margin=dict(t=50, b=10))
st.plotly_chart(fig, width="stretch", theme=None)
best = mean.loc[mean["Balanced acc."].idxmax()]
st.success(f"Best model by balanced accuracy: **{best.Model}** (accuracy {best.Accuracy:.3f}, "
           f"balanced accuracy {best['Balanced acc.']:.3f}, AUC {best.AUC:.3f}). "
           "Continue to **⚖️ Fairness Audit** to check whether this performance is shared equally by all groups.")
