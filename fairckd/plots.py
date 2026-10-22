"""Publication-quality figures (matplotlib, 300 dpi) used by the CLI pipeline and PDF report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
GROUP_COLORS = ["#0F766E", "#9F1D35", "#C27C0E", "#4F46E5", "#64748B", "#0E7490", "#7C3AED", "#B45309",
                "#15803D", "#BE185D"]
METHOD_COLORS = {"Baseline": "#94A3B8", "Unawareness": "#C9A227", "Reweighing": "#4F46E5",
                 "Exp. Gradient": "#0E7490", "Adversarial": "#15803D", "Threshold Opt.": "#9F1D35", "RW + TO (hybrid)": "#7C3AED",
                 "SMOTE": "#C2410C", "Reject Option": "#BE185D"}


def _wrap(s, n=12):
    return s.replace(" ", "\n", 1) if len(s) > n else s


def group_summary(summary: pd.DataFrame, title=""):
    d = summary[summary.Group != "All"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = GROUP_COLORS[:len(d)]
    ax[0].bar(d.Group.map(_wrap), d.Patients, color=cols); ax[0].set_title("Patients per group"); ax[0].set_ylabel("Patients")
    ax[1].bar(d.Group.map(_wrap), d["CKD prevalence (%)"], color=cols); ax[1].set_title("CKD prevalence per group")
    ax[1].set_ylabel("%")
    for i, v in enumerate(d["CKD prevalence (%)"]):
        ax[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    fig.suptitle(title, y=1.02)
    return fig


def feature_ranking(rank: pd.DataFrame, highlight=()):
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(rank) + 1))
    c = ["#C0504D" if f in highlight else "#4F81BD" for f in rank.Feature]
    ax.barh(rank.Description[::-1], rank["Mutual information"][::-1], color=c[::-1])
    ax.set_xlabel("Mutual information with CKD"); ax.set_title("Feature ranking")
    return fig


def performance(perf: pd.DataFrame, title=""):
    """perf: index = model, columns = metrics (means)."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    perf.plot(kind="bar", ax=ax, rot=0, color=["#4F81BD", "#9BBB59", "#F79646", "#8064A2"][:perf.shape[1]])
    ax.set_xticklabels([_wrap(t.get_text()) for t in ax.get_xticklabels()])
    ax.set_ylim(max(0, perf.values.min() - 0.1), 1.01); ax.set_title(title or "Model performance"); ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def group_metric(rates: pd.DataFrame, metric="TPR", title=""):
    """rates: index = model, columns = groups."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rates.plot(kind="bar", ax=ax, rot=0, color=GROUP_COLORS[:rates.shape[1]])
    ax.set_xticklabels([_wrap(t.get_text()) for t in ax.get_xticklabels()])
    lo = np.nanmin(rates.values)
    ax.set_ylim(max(0, lo - 0.15), min(1.02, np.nanmax(rates.values) + 0.08))
    ax.set_ylabel({"TPR": "Sensitivity (TPR)", "FPR": "False positive rate"}.get(metric, metric))
    ax.set_title(title); ax.set_xlabel(""); ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    return fig


def fairness_heatmap(fair: pd.DataFrame, title=""):
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(fair) + 1.2))
    sns.heatmap(fair, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=max(0.3, float(np.nanmax(fair.values))), ax=ax)
    ax.set_title(title or "Fairness gaps (lower = fairer)"); ax.set_ylabel("")
    return fig


def mitigation(mm: pd.DataFrame, title=""):
    """mm columns: Model, Method(short), Accuracy/Balanced acc., EOpp diff, EOdds diff."""
    mm = mm.copy(); mm["Model"] = mm.Model.map(_wrap)
    acc = "Balanced acc." if "Balanced acc." in mm else "Accuracy"
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, met in enumerate(["EOpp diff", "EOdds diff", acc]):
        sns.barplot(data=mm, x="Model", y=met, hue="Method", ax=ax[i], palette=METHOD_COLORS)
        ax[i].set_title(met + (" (lower = fairer)" if "diff" in met else "")); ax[i].set_xlabel("")
        if "diff" not in met:
            ax[i].set_ylim(max(0, mm[met].min() - 0.08), min(1, mm[met].max() + 0.03))
        if i < 2:
            ax[i].get_legend().remove()
    ax[2].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, y=1.02)
    return fig


def tradeoff(mm: pd.DataFrame, title=""):
    acc = "Balanced acc." if "Balanced acc." in mm else "Accuracy"
    from matplotlib.lines import Line2D
    markers = dict(zip(mm.Model.unique(), "os^Dv"))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for _, r in mm.iterrows():
        ax.scatter(r["EOdds diff"], r[acc], marker=markers[r.Model], s=90, color=METHOD_COLORS[r.Method], edgecolor="k")
    h = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", label=k) for k, c in METHOD_COLORS.items()
         if k in set(mm.Method)] + [Line2D([], [], marker=v, ls="", color="grey", label=k) for k, v in markers.items()]
    ax.legend(handles=h, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlabel("Equalized Odds difference  (← fairer)"); ax.set_ylabel(f"{acc}  (↑ better)")
    ax.set_title(title or "Fairness–accuracy trade-off")
    return fig


def representation(rep: pd.DataFrame, group, other):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for m, ls in zip(rep.Model.unique(), ["-", "--", ":"]):
        b = rep[(rep.Model == m) & (rep.Method == "Baseline")]
        ax[0].plot(b["Kept (%)"], b[f"Accuracy [{group}]"], ls, marker="o", color="#4F81BD", label=f"{m}: {group}")
        ax[0].plot(b["Kept (%)"], b[f"Accuracy [{other}]"], ls, marker="o", color="#C0504D", label=f"{m}: {other}")
        for meth, c in [("Baseline", "#7F7F7F"), ("Reweighing", "#9BBB59")]:
            d = rep[(rep.Model == m) & (rep.Method == meth)]
            ax[1].plot(d["Kept (%)"], d["EOdds diff"], ls, marker="o", color=c, label=f"{m}: {meth}")
    for a in ax:
        a.invert_xaxis(); a.set_xlabel(f"% of '{group}' patients kept in training data"); a.legend(fontsize=7)
    ax[0].set_ylabel("Accuracy within group"); ax[0].set_title("Group accuracy under representation bias")
    ax[1].set_ylabel("Equalized Odds difference"); ax[1].set_title("Fairness gap vs. under-representation")
    return fig


def shap_summary(sv, title=""):
    plt.figure(); shap.plots.beeswarm(sv, max_display=12, show=False); plt.title(title)
    return plt.gcf()


def shap_by_group(imp: pd.DataFrame, title="", top=12):
    imp = imp.drop(columns="Overall").head(top)[::-1]
    groups = list(imp.columns)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(imp) * max(1, len(groups) / 2) + 1))
    h = 0.8 / len(groups); yy = np.arange(len(imp))
    for i, g in enumerate(groups):
        ax.barh(yy + (i - (len(groups) - 1) / 2) * h, imp[g], h, color=GROUP_COLORS[i % 6], label=g)
    ax.set_yticks(yy); ax.set_yticklabels(imp.index); ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title or "Feature reliance by group"); ax.legend(fontsize=8, loc="lower right")
    return fig


def shap_waterfall(sv_row, title=""):
    plt.figure(); shap.plots.waterfall(sv_row, max_display=10, show=False); plt.title(title)
    return plt.gcf()


def label_bias(lb: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    lb[["CKD % (2021 race-free eGFR)", "CKD % (2009 race-based eGFR)"]].plot(kind="bar", ax=ax, rot=0,
                                                                             color=["#4F81BD", "#C0504D"])
    ax.set_xticklabels([_wrap(t.get_text(), 10) for t in ax.get_xticklabels()])
    ax.set_ylabel("CKD prevalence (%)"); ax.set_xlabel(""); ax.legend(fontsize=8)
    ax.set_title("Label bias: CKD labels under race-based vs race-free eGFR equation")
    return fig


def calibration(curves: pd.DataFrame, title=""):
    models = list(curves.Model.unique())
    fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4), squeeze=False)
    for i, m in enumerate(models):
        a = ax[0][i]; d = curves[curves.Model == m]
        a.plot([0, 1], [0, 1], ":", color="grey", lw=1)
        for k, g in enumerate(sorted(d.Group.unique())):
            dg = d[(d.Group == g) & (d["n"] >= 5)]
            a.plot(dg["Mean predicted"], dg["Observed CKD rate"], "o-", ms=4, color=GROUP_COLORS[k % 10], label=g)
        a.set_title(f"{m}"); a.set_xlabel("Mean predicted probability"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        if i == 0:
            a.set_ylabel("Observed CKD rate")
        a.legend(fontsize=7)
    fig.suptitle(title, y=1.02)
    return fig


def pareto_select(pareto: pd.DataFrame, select_raw: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    mk = {"Logistic Regression": "o", "Random Forest": "s", "XGBoost": "^"}
    for (model, meth), r in pareto.iterrows():
        sm = meth if meth in METHOD_COLORS else {"Baseline": "Baseline"}.get(meth, meth)
        from .mitigation import SHORT
        lab = SHORT.get(meth, meth)
        ax.scatter(r["EOdds diff"], r["Balanced acc."], marker=mk.get(model, "o"), s=80,
                   color=METHOD_COLORS.get(lab, "grey"), edgecolor="black" if r["Pareto-optimal"] else "none",
                   linewidth=1.4, zorder=3)
    front = pareto[pareto["Pareto-optimal"]].sort_values("EOdds diff")
    ax.plot(front["EOdds diff"], front["Balanced acc."], "--", color="black", lw=1, zorder=2, label="Pareto front")
    sel = select_raw.groupby("Strategy").mean(numeric_only=True)
    for name, c, m in [("FairCKD-Select", "#0F766E", "*"), ("Accuracy-driven selection", "#9F1D35", "X")]:
        if name in sel.index:
            ax.scatter(sel.loc[name, "EOdds diff"], sel.loc[name, "Balanced acc."], marker=m, s=260, color=c,
                       edgecolor="black", zorder=4, label=f"{name} (nested CV)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in METHOD_COLORS.items()] + \
        [Line2D([], [], marker=v, ls="", color="grey", label=k) for k, v in mk.items()]
    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(handles=h + h2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_xlabel("Equalized Odds difference (lower is fairer)"); ax.set_ylabel("Balanced accuracy")
    ax.set_title("Fairness-accuracy Pareto front and FairCKD-Select")
    return fig


def external(ext: pd.DataFrame, groups):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    d = ext.copy(); d["Label"] = d.Model.str.replace("Logistic Regression", "LR") + "\n" + d.Method
    x = np.arange(len(d))
    ax[0].bar(x - 0.2, d["Balanced acc."], 0.4, color="#0F766E", label="Balanced accuracy")
    ax[0].bar(x + 0.2, d["AUC"].fillna(0), 0.4, color="#64748B", label="AUC")
    ax[0].set_ylim(0.5, 0.9); ax[0].set_xticks(x); ax[0].set_xticklabels(d.Label, fontsize=6, rotation=60, ha="right")
    ax[0].set_title("External validation: performance"); ax[0].legend(fontsize=7)
    ax[1].bar(x, d["EOdds diff"], color=[METHOD_COLORS.get(m, "grey") for m in d.Method])
    ax[1].set_xticks(x); ax[1].set_xticklabels(d.Label, fontsize=6, rotation=60, ha="right")
    ax[1].set_title("External validation: Equalized Odds gap (lower is fairer)")
    return fig


def proxy(features: pd.DataFrame, auc: pd.DataFrame, top=10):
    """Left: features that differ most between groups (standardised mean difference).
    Right: how well the group can be recovered from the features (cross-validated AUC)."""
    d = features.sort_values("Max standardised mean difference", ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(d) + 1.6), gridspec_kw={"width_ratios": [1.4, 1]})
    c = ["#9F1D35" if v == "Yes" else "#94A3B8" for v in d["Potential proxy"]]
    ax[0].barh(d.Description, d["Max standardised mean difference"], color=c)
    ax[0].axvline(0.5, ls=":", color="#14213D"); ax[0].set_xlabel("Largest standardised mean difference between groups")
    ax[0].set_title("Features that differ between groups (red = potential proxy, ≥ 0.5)")
    a = auc.set_index("Feature set")
    x = np.arange(len(a))
    ax[1].bar(x - 0.18, a["AUC (Logistic Regression)"], 0.36, color="#0F766E", label="Logistic Regression")
    ax[1].bar(x + 0.18, a["AUC (XGBoost)"], 0.36, color="#64748B", label="XGBoost")
    for i, (u, v) in enumerate(zip(a["AUC (Logistic Regression)"], a["AUC (XGBoost)"])):
        ax[1].text(i - 0.18, u + 0.01, f"{u:.2f}", ha="center", fontsize=8); ax[1].text(i + 0.18, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].axhline(0.5, ls=":", color="#14213D"); ax[1].set_ylim(0.4, 1.08)
    ax[1].set_xticks(x); ax[1].set_xticklabels([_wrap(s, 18) for s in a.index], fontsize=8)
    ax[1].set_title("Can the group be predicted from the features? (AUC)"); ax[1].legend(fontsize=7, loc="lower right")
    return fig


def pdp_groups(pdp: pd.DataFrame, title=""):
    """Partial dependence per group with ICE curves in the background, one panel per feature."""
    feats = list(dict.fromkeys(pdp.Feature))
    fig, ax = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 3.6), squeeze=False)
    groups = sorted(pdp[pdp.Kind == "Group PDP"].Curve.unique())
    for j, f in enumerate(feats):
        d = pdp[pdp.Feature == f]
        for _, ice in d[d.Kind == "ICE"].groupby("Curve"):
            ax[0, j].plot(ice.Value, ice["CKD probability"], color="#CBD5E1", lw=0.6, zorder=1)
        for i, g in enumerate(groups):
            gd = d[(d.Kind == "Group PDP") & (d.Curve == g)]
            ax[0, j].plot(gd.Value, gd["CKD probability"], color=GROUP_COLORS[i % 10], lw=2.2, label=g, zorder=3)
        al = d[d.Kind == "PDP"]
        ax[0, j].plot(al.Value, al["CKD probability"], color="#14213D", lw=1.6, ls="--", label="All patients", zorder=4)
        ax[0, j].set_title(f); ax[0, j].set_ylim(0, 1); ax[0, j].set_ylabel("Predicted CKD probability" if j == 0 else "")
    ax[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle(title or "Partial dependence by group (lines) with individual ICE curves (grey)", y=1.03)
    return fig


def drift(fairness: pd.DataFrame, data: pd.DataFrame, top=10):
    """Left: Equalized Odds gap internal vs external per pipeline. Right: largest feature shifts (PSI)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
    d = fairness.copy(); d["Label"] = d.Model.str.replace("Logistic Regression", "LR") + "\n" + d.Method
    x = np.arange(len(d))
    ax[0].bar(x - 0.2, d["EOdds diff (internal)"], 0.4, color="#0F766E", label="Development (CV, 2017-18)")
    ax[0].bar(x + 0.2, d["EOdds diff (external)"], 0.4, color="#9F1D35", label="External (2015-16)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(d.Label, fontsize=6, rotation=60, ha="right")
    ax[0].set_title("Fairness drift: Equalized Odds gap (lower is fairer)"); ax[0].legend(fontsize=7, loc="upper right", framealpha=0.95)
    t = data.head(top)[::-1]
    c = ["#9F1D35" if s == "Significant" else "#C27C0E" if s == "Moderate" else "#94A3B8" for s in t.Shift]
    ax[1].barh(t.Feature, t["Max group PSI"], color=c)
    ax[1].axvline(0.1, ls=":", color="#C27C0E"); ax[1].axvline(0.2, ls=":", color="#9F1D35")
    ax[1].set_xlabel("Largest within-group PSI"); ax[1].set_title("Data drift by feature")
    return fig
