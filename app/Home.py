"""FairCKD - Fairness-aware CKD Diagnosis System (Streamlit entry point).

Run from the project folder:   streamlit run app/Home.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st

from common import ASSETS, CSS

st.set_page_config(page_title="FairCKD", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

V = Path(__file__).resolve().parent / "views"
nav = st.navigation({
    "": [st.Page(V / "home.py", title="Overview", icon=":material/space_dashboard:", default=True)],
    "Analysis": [
        st.Page(V / "dataset_analysis.py", title="Dataset analysis", icon=":material/database:"),
        st.Page(V / "model_training.py", title="Model training", icon=":material/model_training:"),
        st.Page(V / "fairness_audit.py", title="Fairness audit", icon=":material/balance:"),
        st.Page(V / "bias_mitigation.py", title="Bias mitigation", icon=":material/tune:"),
        st.Page(V / "explainability.py", title="Explainability", icon=":material/insights:"),
        st.Page(V / "validation.py", title="Validation", icon=":material/verified:"),
    ],
    "Clinic": [
        st.Page(V / "patient_diagnosis.py", title="Patient diagnosis", icon=":material/stethoscope:"),
        st.Page(V / "reports.py", title="Reports", icon=":material/description:"),
    ],
})
nav.run()
