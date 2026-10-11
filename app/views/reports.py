"""Download Excel / PDF fairness-audit reports and figures."""
import io
import zipfile
from pathlib import Path

from common import ROOT, fig_path, page_header, page_setup, saved_results, sidebar, st
from fairckd.pipeline import RESULTS, run_experiment, slug

page_setup("Reports", "📄")
page_header("Reports", None, "PDF fairness-audit reports, Excel result tables and 300-dpi figures for every experiment.")
cfg = sidebar()

st.markdown("### Saved experiments")
runs = sorted(p for p in RESULTS.glob("*/results.joblib"))
if not runs:
    st.info("No saved runs yet. Run `python main.py --all` in the project folder.")
for p in runs:
    d = p.parent
    import joblib
    R = joblib.load(p)
    c = R["config"]
    with st.container(border=True):
        st.markdown(f"**{c['Dataset']}** · {c['Feature set']} · sensitive attribute **{c['Sensitive attribute']}**  \n"
                    f"<span class='small'>{c['Patients']:,} patients · {c['CV protocol']}</span>", unsafe_allow_html=True)
        k = st.columns(3)
        if (d / "report.pdf").exists():
            k[0].download_button("📄 PDF report", (d / "report.pdf").read_bytes(), f"{d.name}_report.pdf",
                                 "application/pdf", key=f"pdf{d.name}")
        if (d / "results.xlsx").exists():
            k[1].download_button("📊 Excel tables", (d / "results.xlsx").read_bytes(), f"{d.name}_results.xlsx",
                                 key=f"xl{d.name}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in (d / "figures").glob("*.png"):
                z.write(f, f.name)
        k[2].download_button("🖼️ Figures (zip)", buf.getvalue(), f"{d.name}_figures.zip", key=f"fg{d.name}")

if cfg:
    spec, fs, at, X, y, A = cfg
    st.markdown("### Generate a report for the current configuration")
    st.caption(f"{spec.name} · {fs} · {at}. Runs all 9 stages (this can take several minutes on large datasets).")
    reps = st.number_input("CV repeats", 1, 10, 1 if len(y) > 2000 else 3)
    if st.button("▶ Run full pipeline and build report", type="primary"):
        log = st.empty(); lines = []

        def show(msg):
            lines.append(msg); log.code("\n".join(lines[-14:]))
        with st.spinner("Running…"):
            out = RESULTS / ("custom__" + slug(spec, fs, at)) if spec.key == "custom" else None
            R = run_experiment(spec, fs, at, repeats_base=int(reps), repeats_mit=int(reps), out_dir=out, log=show)
        st.success("Report ready; it now appears in the list above (refresh the page).")
        st.download_button("📄 Download PDF", Path(R["figures"]["01_group_summary"]).parent.parent.joinpath("report.pdf").read_bytes(),
                           "fairckd_report.pdf", "application/pdf")
