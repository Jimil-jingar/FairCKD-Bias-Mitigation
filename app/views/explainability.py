"""Stage 8: explainable AI with SHAP and LIME."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from common import TEAL, group_colors, insight, np, page_header, page_setup, pd, px, sidebar, st
from fairckd.explain import explain, importance_by_group, lime_explain, lime_figure
from fairckd.models import make_classifier
from fairckd.preprocessing import make_preprocessor

page_setup("Explainability", "🔍")
page_header("Explainability", "Stage 8: explainable AI (SHAP and LIME)",
            "Which clinical features drive the predictions, overall, for each group and for one patient.")
cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg
model_name = st.selectbox("Model to explain", ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"])


@st.cache_resource(show_spinner="Training model and computing SHAP values…")
def shap_for(key, fs, at, model_name):
    pre = make_preprocessor().fit(X)
    m = make_classifier(model_name).fit(pre.transform(X), y)
    sv = explain(m, pre, X, spec.labels)
    return m, pre, sv


model, pre, sv = shap_for(spec.key + str(len(y)), fs, at, model_name)
imp = importance_by_group(sv, A)
groups = sorted(A.unique())
_gi = imp.drop(columns="Overall")
_sp = (_gi.max(axis=1) - _gi.min(axis=1)).sort_values(ascending=False)
_f = _sp.index[0]
insight(imp.index[0],
        f"is what drives {model_name}'s CKD predictions most. The feature it uses most <b>differently between groups</b> is "
        f"<b>{_f}</b>: it weighs {_gi.loc[_f].max():.2f} for <b>{_gi.loc[_f].idxmax()}</b> but {_gi.loc[_f].min():.2f} for "
        f"<b>{_gi.loc[_f].idxmin()}</b>. When the evidence differs by group, so do the errors.",
        bars=[(f, v, "teal" if i == 0 else "slate") for i, (f, v) in enumerate(imp["Overall"].head(5).items())],
        caption="Top 5 features, mean |SHAP value|", fmt="{:.2f}")

t1, t2, t3 = st.tabs(["Global importance", f"By {at.lower()}", "Single patient"])
with t1:
    top = imp.head(15)[::-1]
    f = px.bar(top.reset_index(), x="Overall", y="index", orientation="h", height=30 * len(top) + 120,
               title="Mean |SHAP value| (overall feature importance)", color_discrete_sequence=[TEAL])
    f.update_yaxes(title=""); f.update_xaxes(title="mean |SHAP value|"); f.update_layout(margin=dict(t=50, b=10))
    st.plotly_chart(f, width="stretch", theme=None)
    with st.expander("SHAP beeswarm plot (direction of each feature's effect)"):
        plt.figure(); shap.plots.beeswarm(sv, max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Each dot is a patient. Red = high feature value, blue = low. Right of 0 = pushes towards CKD.")

with t2:
    long = imp.head(12).drop(columns="Overall").reset_index().melt("index", var_name="Group", value_name="mean |SHAP|")
    f = px.bar(long, y="index", x="mean |SHAP|", color="Group", barmode="group", orientation="h",
               height=55 * 12 + 100, title=f"Feature reliance by {at.lower()}", color_discrete_map=group_colors(groups))
    f.update_yaxes(title="", autorange="reversed"); f.update_layout(margin=dict(t=50, b=10), legend_title_text="")
    st.plotly_chart(f, width="stretch", theme=None)
    gi = imp.drop(columns="Overall")
    spread = (gi.max(axis=1) - gi.min(axis=1)).sort_values(ascending=False).head(3)
    st.info("Largest differences in how the model uses a feature between groups: " +
            ", ".join(f"**{k}** (Δ {v:.2f})" for k, v in spread.items()) +
            ". When the model relies on different evidence for different groups, its errors also differ: "
            "this is the mechanism behind the bias found in the Fairness Audit.")

with t3:
    c = st.columns([1, 1, 2])
    g = c[0].selectbox("Group", groups)
    lab = c[1].selectbox("True status", ["CKD", "Not CKD"])
    idx = np.where((A.values == g) & (y.values == (1 if lab == "CKD" else 0)))[0]
    if len(idx) == 0:
        st.warning("No patient in this subgroup."); st.stop()
    i = c[2].select_slider("Patient", options=list(idx), value=int(idx[0]))
    p = float(model.predict_proba(pre.transform(X.iloc[[i]]))[0, 1])
    st.markdown(f"Patient **#{i}** · group *{g}* · true status **{lab}** · predicted probability of CKD **{100 * p:.1f}%**")
    l1, l2 = st.columns(2, gap="large")
    with l1:
        st.markdown("**SHAP**: exact contribution of each feature")
        plt.figure(); shap.plots.waterfall(sv[i], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Red bars push the prediction towards CKD, blue bars away from CKD (log-odds scale for tree models).")
    with l2:
        st.markdown("**LIME**: a simple local model fitted around this patient")
        lt, linfo = lime_explain(model, pre, X.iloc[[i]], spec.labels, [c for c in spec.binary if c in X.columns])
        st.pyplot(lime_figure(lt), clear_figure=True)
        st.caption(f"Red pushes towards CKD, blue away (probability scale). Local fit R² = {linfo['local_r2']:.2f}.")
    shap_top = [sv.feature_names[k] for k in np.argsort(-np.abs(sv[i].values))[:5]]
    common5 = [f for f in lt.Feature.head(5) if f in shap_top]
    st.info(f"SHAP and LIME agree on **{len(common5)} of the top 5** features for this patient"
            + (f": {', '.join(common5)}." if common5 else ".")
            + " Two different explanation methods pointing to the same features makes the explanation more trustworthy.")
