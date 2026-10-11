"""Stage 8 - Explainable AI.

* SHAP (Lundberg & Lee, 2017): TreeExplainer for tree models, LinearExplainer for logistic regression.
* LIME (Ribeiro, Singh & Guestrin, 2016): a weighted linear surrogate fitted on perturbed copies of one patient.
  Implemented here directly (tabular, continuous mode) so the app needs no extra package.
"""
import numpy as np
import pandas as pd
import shap


def explain(model, preprocessor, X: pd.DataFrame, labels: dict | None = None):
    """Return a shap.Explanation whose .data holds the (imputed, unscaled) clinical values."""
    labels = labels or {}
    Z = preprocessor.transform(X)
    if hasattr(model, "get_booster") or hasattr(model, "estimators_") or hasattr(model, "tree_"):
        ex = shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        ex = shap.LinearExplainer(model, Z)
    else:
        bg = shap.sample(Z, min(100, len(Z)), random_state=0)
        ex = shap.Explainer(lambda z: model.predict_proba(z)[:, 1], bg)
    sv = ex(Z)
    if sv.values.ndim == 3:                       # (n, features, classes) -> CKD class
        sv = sv[:, :, 1]
    sv.data = preprocessor.named_steps["impute"].transform(X)
    sv.feature_names = [labels.get(c, c) for c in X.columns]
    return sv


def importance_by_group(sv, A) -> pd.DataFrame:
    """Mean |SHAP| per feature for each sensitive group - does the model rely on different evidence per group?"""
    df = pd.DataFrame(np.abs(sv.values), columns=sv.feature_names)
    imp = df.groupby(np.asarray(A)).mean().T
    imp["Overall"] = df.mean()
    return imp.sort_values("Overall", ascending=False).round(4)


def lime_explain(model, preprocessor, x: pd.DataFrame, labels: dict | None = None, binary=(),
                 n_samples: int = 3000, kernel_width: float | None = None, seed: int = 0):
    """LIME explanation for one patient (a 1-row DataFrame).

    Perturbed patients are drawn from the training distribution in the standardised feature space
    (normal for continuous features, Bernoulli at the training frequency for yes/no features), the model
    scores them, and a ridge regression weighted by an exponential kernel on the distance to the patient
    is fitted. Each feature's contribution is its surrogate weight times the patient's standardised value,
    i.e. how much that value moves the CKD probability away from an average patient.

    Returns (DataFrame with Feature, Value, LIME weight, Contribution; sorted by |Contribution|), info dict.
    """
    from sklearn.linear_model import Ridge
    labels = labels or {}
    z0 = preprocessor.transform(x)[0]
    sc = preprocessor.named_steps["scale"]
    d = len(z0)
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_samples, d))
    for j, c in enumerate(x.columns):
        if c in binary:
            raw = (rng.random(n_samples) < np.clip(sc.mean_[j], 0, 1)).astype(float)
            Z[:, j] = (raw - sc.mean_[j]) / sc.scale_[j]
    Z[0] = z0
    target = model.predict_proba(Z)[:, 1]
    width = kernel_width or 0.75 * np.sqrt(d)
    dist = np.sqrt(((Z - z0) ** 2).sum(axis=1))
    w = np.sqrt(np.exp(-(dist ** 2) / width ** 2))
    sur = Ridge(alpha=1.0).fit(Z, target, sample_weight=w)
    raw_vals = preprocessor.named_steps["impute"].transform(x)[0]
    out = pd.DataFrame({"Feature": [labels.get(c, c) for c in x.columns], "Value": raw_vals,
                        "LIME weight": sur.coef_, "Contribution": sur.coef_ * z0})
    out = out.reindex(out.Contribution.abs().sort_values(ascending=False).index).reset_index(drop=True)
    info = {"intercept": float(sur.intercept_), "local_r2": float(sur.score(Z, target, sample_weight=w)),
            "surrogate_prediction": float(sur.predict(z0[None])[0]), "model_prediction": float(target[0])}
    return out, info


def lime_figure(table: pd.DataFrame, top: int = 10, title: str = "Why this prediction? (LIME)"):
    """Horizontal bar chart (matplotlib) of the largest LIME contributions."""
    import matplotlib.pyplot as plt
    t = table.head(top).iloc[::-1]
    lab = [f"{f} = {v:g}" for f, v in zip(t.Feature, np.round(t.Value, 2))]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.2))
    ax.barh(lab, t.Contribution, color=["#E0566F" if v > 0 else "#2B7BD6" for v in t.Contribution])
    ax.axvline(0, color="#14213D", lw=0.8)
    ax.set_xlabel("Change in CKD probability vs. an average patient")
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
