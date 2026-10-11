"""Shared helpers for the FairCKD Streamlit app: sidebar configuration, caching, styling, charts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fairckd import data as D
from fairckd.pipeline import RESULTS, slug

# ----------------------------------------------------------------------------- design tokens
INK, TEAL, MAROON, PAPER, LINE, MUTED = "#14213D", "#0F766E", "#9F1D35", "#F5F7F9", "#DCE3EA", "#5B6B7F"
# one fixed colour per group everywhere in the app (sorted group order)
GROUP_COLORS = [TEAL, MAROON, "#C27C0E", "#4F46E5", "#64748B", "#0E7490"]
METHOD_COLORS = {"Baseline": "#94A3B8", "Unawareness": "#C9A227", "Reweighing": "#4F46E5",
                 "Exp. Gradient": "#0E7490", "Threshold Opt.": MAROON, "RW + TO (hybrid)": "#7C3AED"}
SEQ = [TEAL, "#4F46E5", "#C27C0E", MAROON]

import plotly.io as pio
pio.templates["fairckd"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, Segoe UI, Arial, sans-serif", color=INK, size=13),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    colorway=GROUP_COLORS, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#E8EDF2", linecolor=LINE, zerolinecolor=LINE, ticks="", tickangle=0, automargin=True),
    yaxis=dict(gridcolor="#E8EDF2", linecolor=LINE, zerolinecolor=LINE, ticks=""),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK)),
    bargap=0.25))
pio.templates.default = "plotly_white+fairckd"


def wrap(label: str) -> str:
    """Two-line axis label for long model names, so ticks stay horizontal."""
    w = str(label).split(" ")
    return label if len(w) < 2 or len(label) < 10 else " ".join(w[:len(w) // 2 or 1]) + "<br>" + " ".join(w[len(w) // 2 or 1:])


def group_colors(groups):
    """Stable colour per group: same group, same colour on every page."""
    return {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(sorted(groups))}


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif !important; }
.stApp {background: #F5F7F9;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton, [data-testid="stAppDeployButton"],
#MainMenu, footer {display: none !important;}
h1, h2, h3, h4 {color: #14213D; letter-spacing: -0.01em;}
h2 {font-weight: 600 !important; font-size: 1.35rem !important; margin-top: 1.2rem !important;}
h3 {font-weight: 600 !important; font-size: 1.1rem !important;}
p, li {line-height: 1.6;}

/* sidebar */
section[data-testid="stSidebar"] {background: #14213D;}
section[data-testid="stSidebar"] * {color: #DCE6F2;}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {border-radius: 6px; padding: 0.35rem 0.6rem;}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {background: rgba(255,255,255,0.06);}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {background: #0F766E;}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {color: #fff !important; font-weight: 600;}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {background: #FFFFFF !important; border-color: #FFFFFF !important; border-radius: 6px;}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input, section[data-testid="stSidebar"] [data-testid="stSelectbox"] button
  {background: #FFFFFF !important; color: #14213D !important;}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {color: #14213D !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {background: #FFFFFF !important; border-color: #FFFFFF !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] * {color: #14213D !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {fill: #14213D !important;}
section[data-testid="stSidebar"] label p {color: #9FB3CC !important; font-size: 0.82rem !important;}
section[data-testid="stSidebar"] hr {border-color: #2E4170;}
.side-title {font-size: 0.95rem; font-weight: 600; color: #fff !important; margin: 0.4rem 0 0.2rem;}
.side-meta {font-size: 0.8rem; color: #9FB3CC !important; line-height: 1.5;}

/* page header */
.ph {padding: 0 0 1.1rem; margin-bottom: 1.2rem; border-bottom: 1px solid #DCE3EA;}
.ph-stage {display: inline-block; font-size: 0.78rem; font-weight: 600; color: #0F766E; background: #E3F2F0;
  border-radius: 4px; padding: 2px 8px; margin-bottom: 0.5rem;}
.ph h1 {font-size: 2.1rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.15;}
.ph p {color: #5B6B7F; margin: 0.45rem 0 0; font-size: 1rem; max-width: 72ch;}

/* metrics */
[data-testid="stMetric"] {background: #fff; border: 1px solid #DCE3EA; border-radius: 8px; padding: 0.8rem 1rem;}
[data-testid="stMetricLabel"] p {color: #5B6B7F !important; font-size: 0.85rem !important;}
[data-testid="stMetricValue"] {color: #14213D; font-weight: 600;}

/* buttons, tabs, containers */
.stButton > button[kind="primary"], .stFormSubmitButton > button, .stDownloadButton > button[kind="primary"] {
  background: #0F766E; border: 1px solid #0F766E; color: #fff; font-weight: 600; border-radius: 6px;}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {background: #0B5F59; border-color: #0B5F59;}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem; border-bottom: 1px solid #DCE3EA;}
.stTabs [data-baseweb="tab"] {font-weight: 500;}
.stTabs [aria-selected="true"] {color: #0F766E !important;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {background: #fff; border: 1px solid #DCE3EA; border-radius: 10px; padding: 6px; box-sizing: border-box; max-width: 100%;}
[data-testid="stForm"] {background: #fff; border: 1px solid #DCE3EA; border-radius: 10px;}
[data-testid="stExpander"] {background: #fff; border-radius: 8px;}
.main [data-testid="stNumberInputContainer"], .main [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-testid="stNumberInputContainer"], [data-testid="stForm"] [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-testid="stSelectbox"] [role="group"] {background: #F1F4F7 !important; border: 1px solid #DCE3EA !important; border-radius: 6px;}
[data-testid="stForm"] input {background: transparent !important;}
.small {font-size: 0.85rem; color: #5B6B7F;}

/* home: hero */
.hero {background: #14213D; border-radius: 14px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.6rem;
  display: grid; grid-template-columns: 1.15fr 1fr; gap: 2.4rem; align-items: center;}
.hero h1 {color: #fff; font-size: 2.6rem; font-weight: 700; line-height: 1.08; margin: 0 0 0.9rem; letter-spacing: -0.02em;}
.hero p {color: #C9D6E6; font-size: 1.02rem; margin: 0; max-width: 52ch;}
.hero .who {color: #8FA6C2; font-size: 0.85rem; margin-top: 1.2rem; line-height: 1.55;}
.gap-card {background: #1C2C50; border-radius: 10px; padding: 1.3rem 1.4rem;}
.gap-card .q {color: #C9D6E6; font-size: 0.9rem; margin-bottom: 1rem;}
.gap-row {margin-bottom: 1rem;}
.gap-row .lbl {display: flex; justify-content: space-between; font-size: 0.92rem; color: #fff; margin-bottom: 0.35rem;}
.gap-row .lbl b {font-size: 1.5rem; font-weight: 700;}
.gap-track {background: #2A3B63; border-radius: 4px; height: 14px; overflow: hidden;}
.gap-fill {height: 100%; border-radius: 4px; animation: grow 1.4s ease-out both;}
@keyframes grow {from {width: 0;}}
@media (prefers-reduced-motion: reduce) {.gap-fill {animation: none;}}
.gap-card .src {color: #8FA6C2; font-size: 0.78rem; margin-top: 0.4rem;}
@media (max-width: 900px) {.hero {grid-template-columns: 1fr; padding: 1.6rem;} .hero h1 {font-size: 2rem;}}

/* home: pipeline */
.pipe {display: grid; grid-template-columns: repeat(9, 1fr); gap: 6px; margin: 0.6rem 0 0.4rem;}
.pipe div {background: #fff; border: 1px solid #DCE3EA; border-top: 3px solid #0F766E; border-radius: 6px;
  padding: 0.6rem 0.55rem; font-size: 0.8rem; color: #14213D; line-height: 1.35; min-height: 92px;}
.pipe div.fair {border-top-color: #9F1D35;}
.pipe b {display: block; font-size: 1.1rem; color: #0F766E; margin-bottom: 0.15rem;}
.pipe div.fair b {color: #9F1D35;}
@media (max-width: 900px) {.pipe {grid-template-columns: repeat(3, 1fr);}}

/* patient result card */
.rx {background: #fff; border: 1px solid #DCE3EA; border-radius: 10px; overflow: hidden;}
.rx-head {padding: 1rem 1.2rem; color: #fff;}
.rx-head.pos {background: #9F1D35;} .rx-head.neg {background: #0F766E;}
.rx-head .big {font-size: 2.6rem; font-weight: 700; line-height: 1;}
.rx-head .lab {font-size: 0.95rem; opacity: 0.92; margin-top: 0.3rem;}
.rx-body {padding: 0.9rem 1.2rem; font-size: 0.92rem; color: #14213D;}
.rx-body div {display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px solid #EEF2F6;}
.rx-body div:last-child {border-bottom: none;}
.rx-body span {color: #5B6B7F;}
.rx-note {padding: 0.8rem 1.2rem; background: #F5F7F9; font-size: 0.88rem; color: #14213D;}
/* insight banner: the headline finding of each page */
.ins {background: #14213D; border-radius: 12px; padding: 1.5rem 1.8rem; color: #fff; margin: 0 0 1.4rem;
  display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center;}
.ins .big {font-size: 2.5rem; font-weight: 700; line-height: 1.05; letter-spacing: -0.02em; color: #fff;}
.ins .txt {color: #C9D6E6; font-size: 0.98rem; margin-top: 0.6rem; line-height: 1.55; max-width: 56ch;}
.ins .txt b {color: #fff;}
.ins .bars {background: #1C2C50; border-radius: 10px; padding: 1rem 1.2rem;}
.ins .bars .cap {color: #9FB3CC; font-size: 0.82rem; margin-bottom: 0.7rem;}
.ins .br {margin-bottom: 0.7rem;} .ins .br:last-child {margin-bottom: 0;}
.ins .br .l {display: flex; justify-content: space-between; font-size: 0.86rem; color: #E6EDF6; margin-bottom: 0.25rem;}
.ins .br .l b {font-weight: 700; color: #fff;}
.ins .br .t {background: #2A3B63; border-radius: 4px; height: 10px; overflow: hidden;}
.ins .br .f {height: 100%; border-radius: 4px; animation: grow 1.2s ease-out both;}
@media (prefers-reduced-motion: reduce) {.ins .br .f {animation: none;}}
.xt {background: white; border: 1px solid #E2E8F0; border-radius: 12px; overflow-x: auto; margin: 0.3rem 0 0.6rem;}
.xt table {width: 100%; border-collapse: collapse; font-size: 0.86rem;}
.xt th {background: #F1F5F9; color: #475569; font-weight: 600; font-size: 0.76rem; text-transform: uppercase;
  letter-spacing: 0.03em; padding: 10px 10px; text-align: right; vertical-align: bottom; line-height: 1.25;}
.xt th.hl {background: #E6F4F1; color: #0F766E;}
.xt td {padding: 9px 10px; border-top: 1px solid #EEF2F6; text-align: right; color: #1E293B; white-space: nowrap;}
.xt .l {text-align: left;}
.xt tr:hover td {background: #F8FAFC;}
.xt .arr {color: #94A3B8;}
.xt .gbar {display: inline-block; width: 46px; height: 6px; background: #E2E8F0; border-radius: 3px;
  margin-right: 7px; vertical-align: middle; overflow: hidden;}
.xt .gbar span {display: block; height: 100%; background: #0F766E; border-radius: 3px;}
@media (max-width: 900px) {.ins {grid-template-columns: 1fr;}}
/* sidebar nav section headers */
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {color: #7F93AE !important; font-size: 0.78rem;}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {color: #7FD3C7 !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {color: #fff !important;}
</style>
"""


DATASETS = {"UCI Chronic Kidney Disease (India)": "uci", "NHANES 2017-18 (USA)": "nhanes", "Upload your own CSV": "custom"}


ASSETS = Path(__file__).resolve().parent / "assets"


def page_setup(title=None, icon=None):
    """Page config, CSS and logo are set once by the router (app/Home.py)."""
    return None


def page_header(title, stage=None, text=None):
    """Consistent page header: framework stage tag, title, one-line purpose."""
    tag = f"<div class='ph-stage'>{stage}</div>" if stage else ""
    sub = f"<p>{text}</p>" if text else ""
    st.markdown(f"<div class='ph'>{tag}<h1>{title}</h1>{sub}</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading dataset…")
def get_spec(key):
    return D.load(key)


def sidebar():
    """Dataset / feature-set / sensitive-attribute selector shared by every page."""
    ss = st.session_state
    with st.sidebar:
        st.markdown("<div class='side-title'>Experiment setup</div>", unsafe_allow_html=True)
        names = list(DATASETS)
        cur = ss.get("ds_name", names[0])
        ds_name = st.selectbox("Dataset", names, index=names.index(cur) if cur in names else 0)
        ss["ds_name"] = ds_name
        key = DATASETS[ds_name]
        if key == "custom":
            spec = ss.get("custom_spec")
            if spec is None:
                st.info("Upload a CSV on the **Dataset Analysis** page.")
                return None
        else:
            spec = get_spec(key)
        fsets, attrs = list(spec.feature_sets), list(spec.sensitive)
        fs = st.selectbox("Feature set", fsets, index=fsets.index(ss["fs"]) if ss.get("fs") in fsets else 0,
                          format_func=lambda v: v.split(" (")[0])
        at = st.selectbox("Sensitive attribute", attrs, index=attrs.index(ss["attr"]) if ss.get("attr") in attrs else 0)
        ss["fs"], ss["attr"] = fs, at
        X, y, A = spec.get(fs, at)
        saved = saved_results(spec, fs, at)
        st.markdown(f"<div class='side-meta'>{len(y):,} patients, {X.shape[1]} features, {A.nunique()} groups<br>"
                    f"CKD prevalence {100 * y.mean():.1f}%<br>"
                    f"{'Saved thesis results loaded' if saved else 'No saved results: run live'}</div>",
                    unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div class='side-meta'>FairCKD 1.0<br>M.Tech Phase-II, Jimil Jingar<br>"
                    "PIET, Parul University</div>", unsafe_allow_html=True)
    return spec, fs, at, X, y, A


def cfg_key(spec, fs, at):
    return f"{spec.key}|{fs}|{at}"


def saved_results(spec, fs, at):
    if spec.key == "custom":
        return None
    p = RESULTS / slug(spec, fs, at) / "results.joblib"
    if not p.exists():
        return None
    return _load_saved(str(p), p.stat().st_mtime)


@st.cache_resource
def _load_saved(path, mtime):
    return joblib.load(path)


def fig_path(p):
    """Saved figure paths are stored relative to the project root (older runs: absolute)."""
    p = Path(p)
    if not p.is_absolute():
        return str(ROOT / p)
    if p.exists():
        return str(p)
    parts = p.parts
    return str(ROOT.joinpath(*parts[parts.index("results"):])) if "results" in parts else str(p)


def run_store(spec, fs, at):
    runs = st.session_state.setdefault("runs", {})
    return runs.setdefault(cfg_key(spec, fs, at), {})


def short_group(g):
    return g


def bar(df, x, y, color=None, title="", yrange=None, barmode="group", height=380, color_map=None, text=None):
    if color is not None and color_map is None:
        color_map = group_colors(df[color].unique())
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, title=title, height=height, text=text,
                 color_discrete_sequence=[TEAL], color_discrete_map=color_map)
    if color is not None and color == x:          # one bar per category: full width, no redundant legend
        fig.update_layout(barmode="relative", showlegend=False)
    if text is not None:
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    if yrange:
        fig.update_yaxes(range=yrange)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def metric_help():
    from fairckd.fairness import METRIC_INFO
    return pd.DataFrame([{"Metric": k, "Name": v[0], "Meaning": v[1]} for k, v in METRIC_INFO.items()])


def gauge(p, thr):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number={"suffix": "%"},
                                 title={"text": "Probability of CKD"},
                                 gauge={"axis": {"range": [0, 100]},
                                        "bar": {"color": MAROON if p >= thr else TEAL},
                                        "threshold": {"line": {"color": "black", "width": 3}, "value": 100 * thr},
                                        "steps": [{"range": [0, 100 * thr], "color": "#E3F2F0"},
                                                  {"range": [100 * thr, 100], "color": "#F6E4E8"}]}))
    fig.update_layout(height=270, margin=dict(l=40, r=45, t=50, b=10))
    return fig


# ----------------------------------------------------------------------------- page-level headline finding
BAR_TONES = {"teal": "#2BB3A3", "maroon": "#E0566F", "amber": "#E0A23B", "indigo": "#8B85F0", "slate": "#94A3B8"}


def insight(big, text, bars=None, caption="", fmt="{:.0%}"):
    """Navy banner with the page's main finding. bars: list of (label, value 0..1, tone or hex)."""
    right = ""
    if bars:
        vmax = max(max(v for _, v, _ in bars), 1e-9)
        rows = "".join(
            f"<div class='br'><div class='l'><span>{lab}</span><b>{fmt.format(v)}</b></div>"
            f"<div class='t'><div class='f' style='width:{100 * v / vmax if fmt != '{:.0%}' else 100 * v:.0f}%;"
            f"background:{BAR_TONES.get(c, c)}'></div></div></div>" for lab, v, c in bars)
        right = f"<div class='bars'><div class='cap'>{caption}</div>{rows}</div>"
    size = "" if len(str(big)) <= 16 else " style='font-size:1.9rem'"
    st.markdown(f"<div class='ins'><div><div class='big'{size}>{big}</div><div class='txt'>{text}</div></div>{right}</div>",
                unsafe_allow_html=True)


def _lead(v):
    try:
        return float(str(v).split(" ")[0])
    except ValueError:
        return np.nan


def style_table(df, lower_better=(), higher_better=()):
    """Colour cells by their (leading) value: maroon = worse fairness gap, teal = better performance."""
    def shade(col, good_high):
        vals = col.map(_lead)
        lo, hi = np.nanmin(vals.values), np.nanmax(vals.values)
        out = []
        for v in vals:
            if np.isnan(v) or hi - lo < 1e-12:
                out.append(""); continue
            t = (v - lo) / (hi - lo)
            if good_high:
                a = 0.03 + 0.20 * t
                out.append(f"background-color: rgba(15,118,110,{a:.2f})")
            else:
                a = 0.03 + 0.20 * t
                out.append(f"background-color: rgba(159,29,53,{a:.2f})" + ("; font-weight: 600" if t > 0.85 else ""))
        return out
    sty = df.style
    num = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if num:
        sty = sty.format("{:.3f}", subset=num, na_rep="–")
    for c in df.columns:
        if c in lower_better:
            sty = sty.apply(lambda col: shade(col, False), subset=[c])
        elif c in higher_better:
            sty = sty.apply(lambda col: shade(col, True), subset=[c])
    return sty
