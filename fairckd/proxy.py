"""Stage 3b - Proxy-feature audit.

Removing the sensitive attribute from the model inputs ("fairness through unawareness") does not remove group
information if other clinical features encode it. This module measures how much group information the features
carry:

  * per feature: mutual information with the group and the largest standardised mean difference between groups;
  * overall: how well a classifier can recover the patient's group from the clinical features alone
    (5-fold cross-validated AUC, one-vs-rest macro average for more than two groups),
    with and without the features that directly encode the attribute (e.g. age for age group).

AUC = 0.5 means the features carry no group information; AUC near 1 means the group can be read off the
features, so a model can discriminate by group even when the attribute itself is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import SEED

SMD_FLAG = 0.5          # "medium" effect size (Cohen); above this a feature differs clearly between groups


def feature_proxy_table(X: pd.DataFrame, A, labels: dict | None = None) -> pd.DataFrame:
    labels = labels or {}
    A = np.asarray(A).astype(str)
    groups, codes = np.unique(A, return_inverse=True)
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    mi = mutual_info_classif(Xi.values, codes, random_state=SEED)
    rows = []
    for j, c in enumerate(X.columns):
        means = Xi[c].groupby(A).mean()
        sd = Xi[c].std(ddof=1) or 1.0
        smd = float((means.max() - means.min()) / sd)
        rows.append({"Feature": c, "Description": labels.get(c, c), "MI with group": round(float(mi[j]), 4),
                     "Max standardised mean difference": round(smd, 3),
                     "Highest in": means.idxmax(), "Lowest in": means.idxmin(),
                     "Potential proxy": "Yes" if smd >= SMD_FLAG else "No"})
    return pd.DataFrame(rows).sort_values("MI with group", ascending=False).reset_index(drop=True)


def _cv_auc(model, X, codes, n_groups):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    P = cross_val_predict(model, X, codes, cv=skf, method="predict_proba")
    if n_groups == 2:
        return roc_auc_score(codes, P[:, 1])
    return roc_auc_score(codes, P, multi_class="ovr", average="macro")


def group_predictability(X: pd.DataFrame, A, direct_cols=()) -> pd.DataFrame:
    """Cross-validated AUC of recovering the group from the features, with and without direct proxies."""
    A = np.asarray(A).astype(str)
    groups, codes = np.unique(A, return_inverse=True)
    sets = {"All model features": list(X.columns)}
    direct = [c for c in direct_cols if c in X.columns]
    if direct:
        sets[f"Without {', '.join(direct)} (unawareness)"] = [c for c in X.columns if c not in direct]
    rows = []
    for name, cols in sets.items():
        Xi = SimpleImputer(strategy="median").fit_transform(X[cols])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
        xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=SEED, n_jobs=1,
                            verbosity=0)
        rows.append({"Feature set": name, "Features": len(cols),
                     "AUC (Logistic Regression)": round(_cv_auc(lr, Xi, codes, len(groups)), 3),
                     "AUC (XGBoost)": round(_cv_auc(xgb, Xi, codes, len(groups)), 3)})
    return pd.DataFrame(rows)


def proxy_audit(X: pd.DataFrame, A, labels=None, direct_cols=()):
    return feature_proxy_table(X, A, labels), group_predictability(X, A, direct_cols)
