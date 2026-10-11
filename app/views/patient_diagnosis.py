"""Stage 9: clinical decision support for a single patient."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from common import gauge, np, page_header, page_setup, saved_results, sidebar, st
from fairckd.diagnosis import lime_patient, load_model, model_path, predict_patient, train_final_model
from fairckd.explain import lime_figure

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
with st.form("patient"):
    st.markdown("**Patient data** (pre-filled with dataset medians; untick *known* if a test is not available)")
    pid = st.text_input("Patient ID / name (optional, only printed on the PDF report)", "")
    vals, cols = {}, st.columns(4)
    for i, f in enumerate(b["features"]):
        with cols[i % 4]:
            name = spec.label(f)
            if f in b["binary"]:
                opts = BIN.get(f, ("No", "Yes"))
                med = b["medians"][f]
                v = st.selectbox(name, opts, index=int(round(med)) if not np.isnan(med) else 0, key=f"b_{f}")
                vals[f] = float(opts.index(v))
            else:
                lo, hi, med = float(b["mins"][f]), float(b["maxs"][f]), float(b["medians"][f])
                known = True if f == "age" else st.checkbox(f"{name}: known", True, key=f"k_{f}")
                v = st.number_input(name, lo, hi, med, key=f"n_{f}", format="%.3f" if hi < 3 else "%.1f",
                                    label_visibility="visible")
                vals[f] = v if known else np.nan
    submitted = st.form_submit_button("Predict CKD risk", type="primary")

if submitted:
    p, pos, sv = predict_patient(b, vals)
    grp = None
    if at == "Age group" and "age" in vals and not np.isnan(vals["age"]):
        grp = "Elderly (>=60)" if vals["age"] >= 60 else "Non-elderly (<60)"
    elif at == "Sex" and "female" in vals:
        grp = "Female" if vals["female"] >= 0.5 else "Male"
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
        plt.figure(); shap.plots.waterfall(sv, max_display=10, show=False)
        plt.title("Why this prediction? (SHAP)")
        png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.gcf().savefig(png, dpi=200, bbox_inches="tight")
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Red bars push the estimate towards CKD, blue bars away from it.")
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
