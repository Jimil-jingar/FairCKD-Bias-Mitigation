"""Stages 1-3: data acquisition, preprocessing overview, feature selection."""
from common import D, GROUP_COLORS, TEAL, bar, insight, page_header, page_setup, pd, px, sidebar, st

page_setup("Dataset Analysis", "📊")
page_header("Dataset analysis", "Stages 1–3: data, preprocessing, feature selection",
            "Who is in the data, how common CKD is in each group, and which clinical features matter most.")

# ---------------------------------------------------------------- custom upload
if st.session_state.get("ds_name") == "Upload your own CSV":
    with st.expander("Upload a CSV dataset", expanded=st.session_state.get("custom_spec") is None):
        up = st.file_uploader("CSV file (one row per patient)", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            st.dataframe(raw.head(), width="stretch")
            c = st.columns(3)
            target = c[0].selectbox("CKD / outcome column", raw.columns)
            positive = c[1].selectbox("Value meaning CKD", sorted(raw[target].dropna().astype(str).unique()))
            sens = c[2].selectbox("Sensitive attribute column", [x for x in raw.columns if x != target])
            if raw[sens].nunique() > 10:
                st.warning("The sensitive column has more than 10 distinct values; please use a categorical column "
                           "(e.g. sex, ethnicity, age group).")
            elif st.button("Use this dataset", type="primary"):
                st.session_state["custom_spec"] = D.load_custom(raw, target, positive, sens, name=up.name)
                st.session_state.pop("fs", None); st.session_state.pop("attr", None)
                st.rerun()

cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg

gs0 = D.group_summary(X, y, A); d0 = gs0[gs0.Group != "All"].sort_values("CKD prevalence (%)", ascending=False)
hi, lo = d0.iloc[0], d0.iloc[-1]
ratio = hi["CKD prevalence (%)"] / max(lo["CKD prevalence (%)"], 1e-9)
insight(f"{ratio:.1f}× more CKD",
        f"CKD is <b>{ratio:.1f} times as common</b> in the <b>{hi.Group}</b> group ({hi['CKD prevalence (%)']:.1f}%) as in "
        f"<b>{lo.Group}</b> ({lo['CKD prevalence (%)']:.1f}%). The smallest group makes up only "
        f"{d0['Share (%)'].min():.1f}% of the data. Unequal base rates and under-representation are where bias starts.",
        bars=[(r.Group, r["CKD prevalence (%)"] / 100, GROUP_COLORS[i % 6]) for i, (_, r) in
              enumerate(d0.sort_values("Group").iterrows())],
        caption=f"CKD prevalence by {at.lower()}")
st.markdown(f"**{spec.name}**: {spec.description}")
c = st.columns(4)
c[0].metric("Patients", f"{len(y):,}"); c[1].metric("Features in set", X.shape[1])
c[2].metric(f"CKD cases ({100 * y.mean():.1f}%)", f"{int(y.sum()):,}"); c[3].metric("Groups", A.nunique())

tab1, tab2, tab3, tab5, tab4 = st.tabs(["Groups & representation", "Missing values", "Feature selection",
                                         "Proxy check", "Data preview"])
with tab1:
    gs = D.group_summary(X, y, A)
    st.dataframe(gs, width="stretch", hide_index=True)
    d = gs[gs.Group != "All"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(bar(d, "Group", "Patients", color="Group", title="Patients per group", height=340), width="stretch", theme=None)
    c2.plotly_chart(bar(d, "Group", "CKD prevalence (%)", color="Group", title="CKD prevalence (base rate) per group",
                        height=340, text="CKD prevalence (%)"), width="stretch", theme=None)
    small = d.loc[d["Share (%)"].idxmin()]
    st.info(f"**Bias sources to watch:** the smallest group (*{small.Group}*) is {small['Share (%)']}% of the data, "
            f"and CKD prevalence ranges from {d['CKD prevalence (%)'].min()}% to {d['CKD prevalence (%)'].max()}% "
            "across groups. Under-representation and unequal base rates are classic sources of algorithmic bias.")

with tab2:
    from fairckd.preprocessing import missing_report
    mr = missing_report(X, A)
    mr.index = [spec.label(i) for i in mr.index]
    st.plotly_chart(bar(mr.reset_index(), "index", "Missing (%)", title="Missing values per feature", height=360)
                    .update_xaxes(title=""), width="stretch", theme=None)
    st.dataframe(mr, width="stretch")
    st.caption("Preprocessing used everywhere in FairCKD: median imputation + standard scaling, "
               "fitted on the training folds only (no data leakage).")

with tab3:
    from fairckd.features import feature_ranking

    @st.cache_data(show_spinner="Ranking features…")
    def rank(key, fs, at):
        return feature_ranking(X, y, spec.labels)
    rk = rank(spec.key, fs, at)
    crit = st.radio("Rank by", ["Mutual information", "|Correlation| with CKD", "RFE rank", "Consensus rank"],
                    horizontal=True)
    asc = crit in ("RFE rank", "Consensus rank")
    r2 = rk.sort_values(crit, ascending=asc).copy()
    xcol = crit
    if asc:                                    # ranks: show a score so longer bar = more important
        xcol = "Score (higher = more important)"
        r2[xcol] = len(r2) + 1 - r2[crit]
    fig = px.bar(r2[::-1], x=xcol, y="Description", orientation="h", height=28 * len(rk) + 120,
                 color_discrete_sequence=[TEAL], hover_data=[crit],
                 title=f"Feature ranking: {crit}")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10)); fig.update_yaxes(title="")
    st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown("**Three methods side by side**: mutual information (non-linear filter), point-biserial correlation "
                "(linear filter) and Recursive Feature Elimination with logistic regression (wrapper). The consensus "
                "rank averages the three. PCA is not used because its components mix all clinical variables, so "
                "explanations would no longer refer to features a clinician can read.")
    top5 = {c: set(rk.sort_values(c, ascending=c in ("RFE rank",)).Description.head(5))
            for c in ["Mutual information", "|Correlation| with CKD", "RFE rank"]}
    agree = set.intersection(*top5.values())
    st.info(f"All three methods put **{len(agree)}** feature(s) in their top 5"
            + (f": {', '.join(sorted(agree))}." if agree else "."))
    st.dataframe(rk.drop(columns="Feature").sort_values("Consensus rank"), width="stretch", hide_index=True)
    st.markdown("**Feature sets available for this dataset:**")
    for k, v in spec.feature_sets.items():
        st.markdown(f"- *{k}*: " + ", ".join(spec.label(f) for f in v))

with tab5:
    from fairckd.proxy import proxy_audit

    @st.cache_data(show_spinner="Checking how much group information the features carry…")
    def proxies(key, fs, at, n):
        return proxy_audit(X, A, spec.labels, spec.proxy_cols.get(at, []))
    pf, pa = proxies(spec.key, fs, at, len(y))
    worst = pa.iloc[-1]
    auc = max(worst["AUC (Logistic Regression)"], worst["AUC (XGBoost)"])
    st.markdown(f"Dropping the sensitive attribute (*fairness through unawareness*) only helps if no other feature "
                f"encodes it. A classifier trained to guess the patient's **{at.lower()}** from "
                f"**{worst['Feature set'].lower()}** reaches **AUC {auc:.2f}** (0.5 = no group information).")
    if auc >= 0.7:
        st.warning(f"The group can still be read from the clinical features (AUC {auc:.2f}), so removing the "
                   "attribute alone cannot make the model blind to it. Mitigation methods that act on error rates "
                   "(Exponentiated Gradient, Adversarial Debiasing, Threshold Optimizer, Reject Option) are needed.")
    st.dataframe(pa, width="stretch", hide_index=True)
    top = pf.sort_values("Max standardised mean difference", ascending=False).head(12)[::-1]
    f = px.bar(top, x="Max standardised mean difference", y="Description", orientation="h",
               color="Potential proxy", color_discrete_map={"Yes": "#9F1D35", "No": "#94A3B8"},
               height=30 * len(top) + 130, title="Features that differ most between groups (standardised mean difference)")
    f.add_vline(x=0.5, line=dict(dash="dot", color="#14213D"))
    f.update_yaxes(title=""); f.update_layout(margin=dict(t=50, b=10), legend_title_text="Potential proxy (≥ 0.5)")
    st.plotly_chart(f, width="stretch", theme=None)
    st.dataframe(pf.drop(columns="Feature"), width="stretch", hide_index=True)

with tab4:
    show = X.copy(); show["CKD"] = y.values; show[at] = A.values
    st.dataframe(show.rename(columns=spec.labels), width="stretch", height=420)
    st.download_button("Download this table (CSV)", show.to_csv(index=False).encode(), "dataset.csv", "text/csv")
