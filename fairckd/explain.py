"""Stage 8 - Explainable AI with SHAP (TreeExplainer for tree models, LinearExplainer for LR)."""
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
