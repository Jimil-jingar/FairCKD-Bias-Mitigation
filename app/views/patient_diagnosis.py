"""Stage 9: clinical decision support for a single patient."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from common import gauge, np, page_header, page_setup, pd, saved_results, sidebar, st
from fairckd.diagnosis import (lime_patient, load_model, model_path, predict_patient, reference_patients,
                               train_final_model)
from fairckd.explain import contribution_figure, lime_figure, probability_contributions

page_setup("Patient Diagnosis", "🩺")
page_header("Patient diagnosis", "Stage 9: clinical decision support",
            "Enter a patient's values to get a CKD risk estimate from the fairness-aware model, with the reasons behind it.")
cfg = sidebar()
if not cfg:
    st.stop()
spec, fs, at, X, y, A = cfg


@st.cache_resource(show_spinner="Loading fairness-aware model…")
def get_bundle(key, fs, at, n):
    p = model_path(spec, fs, at)
    if spec.key != "custom" and p.exists():
        return load_model(p)
    return train_final_model(spec, fs, at, save=spec.key != "custom")


b = get_bundle(spec.key, fs, at, len(y))
st.markdown(f"Model: **{b['model_name']}** trained on {b['n_train']:,} patients · fairness attribute: **{at}** · "
            f"decision threshold: **{b['threshold']:.2f}**")

BIN = {"rbc": ("Normal", "Abnormal"), "pc": ("Normal", "Abnormal"), "pcc": ("Not present", "Present"),
       "ba": ("Not present", "Present"), "appet": ("Good", "Poor"), "female": ("Male", "Female")}
examples = b.get("examples") or reference_patients(X, y)      # older saved models have no examples
numeric = [f for f in b["features"] if f not in b["binary"]]
binary = [f for f in b["features"] if f in b["binary"]]
cfg_id = f"{spec.key}|{fs}"


def load_example(name):
    ex = examples[name]
    for f in numeric:
        lo, hi = float(b["mins"][f]), float(b["maxs"][f])
        st.session_state[f"n_{f}"] = float(min(max(ex[f], lo), hi)) if not np.isnan(ex[f]) else float(b["medians"][f])
        st.session_state[f"u_{f}"] = False
    for f in binary:
        opts = BIN.get(f, ("No", "Yes"))
        st.session_state[f"b_{f}"] = opts[int(round(ex[f])) if not np.isnan(ex[f]) else 0]
    st.session_state["pd_loaded"] = (cfg_id, name)


if st.session_state.get("pd_loaded", (None,))[0] != cfg_id:      # first visit / new dataset: healthy patient
    load_example("healthy")
c = st.columns([1, 1, 3])
if c[0].button("Load typical healthy patient", width="stretch"):
    load_example("healthy")
if c[1].button("Load typical CKD patient", width="stretch"):
    load_example("ckd")
c[2].caption(f"Currently loaded: **{'typical CKD patient' if st.session_state['pd_loaded'][1] == 'ckd' else 'typical healthy patient'}** "
             "(median values of the patients with / without CKD in this dataset). Change any value below.")

with st.form("patient"):
    pid = st.text_input("Patient ID / name (optional, only printed on the PDF report)", "")
    vals = {}
    st.markdown("**Measurements** · tick *not measured* if a test is not available (the value is then imputed)")
    cols = st.columns(4)
    for i, f in enumerate(numeric):
        with cols[i % 4]:
            lo, hi = float(b["mins"][f]), float(b["maxs"][f])
            v = st.number_input(spec.label(f), lo, hi, key=f"n_{f}", format="%.3f" if hi < 3 else "%.1f")
            if f == "age":                       # age is always known: keep the grid aligned with an empty row
                st.markdown("<div style='height:2.475rem'></div>", unsafe_allow_html=True)
                miss = False
            else:
                miss = st.checkbox("not measured", key=f"u_{f}")
            vals[f] = np.nan if miss else v
    if binary:
        st.markdown("**Findings and history**")
        cols = st.columns(4)
        for i, f in enumerate(binary):
            with cols[i % 4]:
                opts = BIN.get(f, ("No", "Yes"))
                v = st.selectbox(spec.label(f), opts, key=f"b_{f}")
                vals[f] = float(opts.index(v))
    submitted = st.form_submit_button("Predict CKD risk", type="primary")

if submitted:
    p, pos, sv = predict_patient(b, vals)
    grp = None
    if at == "Age group" and "age" in vals and not np.isnan(vals["age"]):
        grp = "Elderly (>=60)" if vals["age"] >= 60 else "Non-elderly (<60)"
    elif at == "Sex" and "female" in vals:
        grp = "Female" if vals["female"] >= 0.5 else "Male"
    elif at == "Age × Sex" and {"age", "female"} <= set(vals):
        grp = ("60+" if vals["age"] >= 60 else "<60") + (" Female" if vals["female"] >= 0.5 else " Male")
    saved = saved_results(spec, fs, at)
    val_txt = ""
    if saved is not None:
        r = saved["mitigation_raw"]
        rw = r[(r.Model == "XGBoost") & (r.Method.str.startswith("Reweighing"))].mean(numeric_only=True)
        if len(rw):
            val_txt = (f"<div><span>Validated balanced accuracy</span><b>{rw['Balanced acc.']:.3f}</b></div>"
                       f"<div><span>Validated Equalized Odds gap</span><b>{rw['EOdds diff']:.3f}</b></div>")
    unknown = sum(1 for v in vals.values() if isinstance(v, float) and np.isnan(v))
    advice = ("Confirm with eGFR (serum creatinine) and urine albumin-to-creatinine ratio."
              if pos else "No CKD signal at this threshold. Re-screen if risk factors such as diabetes or hypertension are present.")
    c1, c2 = st.columns([1, 1.55], gap="large")
    with c1:
        st.markdown(f"""
<div class="rx">
  <div class="rx-head {'pos' if pos else 'neg'}">
    <div class="big">{100 * p:.1f}%</div>
    <div class="lab">{'CKD likely' if pos else 'CKD unlikely'}: estimated probability of chronic kidney disease</div>
  </div>
  <div class="rx-body">
    <div><span>Decision threshold</span><b>{100 * b['threshold']:.1f}%</b></div>
    <div><span>Patient group ({at.lower()})</span><b>{grp or 'not determined'}</b></div>
    <div><span>Values imputed (not available)</span><b>{unknown}</b></div>
    <div><span>Model</span><b>{b['model_name']}</b></div>
    {val_txt}
  </div>
  <div class="rx-note">{advice}</div>
</div>""", unsafe_allow_html=True)
        st.write("")
        st.plotly_chart(gauge(p, b["threshold"]), width="stretch", theme=None)
    with c2:
        import tempfile

        import plotly.graph_objects as go
        ct, p0, p1 = probability_contributions(sv, top=10)
        d = ct[::-1]
        lab = [f"{f} = {v:.3g}" if not np.isnan(v) else f for f, v in zip(d.Feature, d.Value)]
        fig = go.Figure(go.Bar(x=d["Contribution (pp)"], y=lab, orientation="h",
                               marker_color=["#9F1D35" if v > 0 else "#0F766E" for v in d["Contribution (pp)"]],
                               text=[f"{v:+.1f}" for v in d["Contribution (pp)"]],
                               textposition=["outside" if v >= 0 else "inside" for v in d["Contribution (pp)"]],
                               insidetextanchor="middle", insidetextfont=dict(color="#FFFFFF"),
                               cliponaxis=False, hovertemplate="%{y}<br>%{x:+.1f} percentage points<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#14213D", width=1))
        fig.update_layout(title="Why this prediction? (SHAP, percentage points)", height=34 * len(d) + 140,
                          xaxis_title=f"Average patient {100 * p0:.1f}% → this patient {100 * p1:.1f}%",
                          margin=dict(t=60, b=10, l=10, r=40))
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
        st.caption("Maroon bars raise the estimated CKD risk, teal bars lower it. The bars add up to the "
                   "difference between this patient's risk and the average patient's risk.")
        mfig = contribution_figure(ct, p0, p1)
        png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        mfig.savefig(png, dpi=200, bbox_inches="tight"); plt.close(mfig)
        with st.expander("Second opinion: LIME explanation"):
            lt, linfo = lime_patient(b, vals)
            st.pyplot(lime_figure(lt), clear_figure=True)
            st.caption(f"LIME fits a simple model around this patient (local fit R² = {linfo['local_r2']:.2f}). "
                       "Red pushes towards CKD, blue away.")
        from fairckd.report import patient_report_pdf
        pdf = patient_report_pdf(b, vals, p, pos, png, patient_id=pid, group=grp)
        st.download_button("Download patient report (PDF)", pdf, f"FairCKD_patient_{pid or 'report'}.pdf",
                           "application/pdf", type="primary")
st.caption("Research prototype, not a medical device. Results must be confirmed by a clinician.")
