"""Stage 3 - Feature selection.

Three complementary rankings of how strongly each clinical feature relates to CKD status:
  * Mutual information (filter, captures non-linear dependence)
  * Absolute point-biserial correlation with CKD (filter, linear)
  * Recursive Feature Elimination with logistic regression (wrapper; Guyon et al., 2002)
plus a consensus rank (mean of the three ranks). PCA is deliberately not used for selection: its components mix
all clinical variables, so SHAP / LIME explanations would no longer refer to features a clinician can read.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import SEED


def feature_ranking(X: pd.DataFrame, y, labels: dict | None = None) -> pd.DataFrame:
    labels = labels or {}
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    yv = np.asarray(y, float)
    mi = mutual_info_classif(Xi, yv.astype(int), random_state=SEED)
    corr = np.array([abs(np.corrcoef(Xi[:, j], yv)[0, 1]) if Xi[:, j].std() > 0 else 0.0
                     for j in range(Xi.shape[1])])
    Zs = StandardScaler().fit_transform(Xi)
    rfe = RFE(LogisticRegression(max_iter=2000, random_state=SEED), n_features_to_select=1, step=1)
    rfe.fit(Zs, yv.astype(int))
    out = pd.DataFrame({"Feature": X.columns, "Description": [labels.get(c, c) for c in X.columns],
                        "Mutual information": mi.round(4), "|Correlation| with CKD": np.nan_to_num(corr).round(4),
                        "RFE rank": rfe.ranking_, "Missing (%)": (100 * X.isna().mean()).round(1).values})
    ranks = pd.DataFrame({"mi": out["Mutual information"].rank(ascending=False),
                          "corr": out["|Correlation| with CKD"].rank(ascending=False),
                          "rfe": out["RFE rank"].rank()})
    out["Consensus rank"] = ranks.mean(axis=1).rank(method="min").astype(int)
    return out.sort_values("Mutual information", ascending=False).reset_index(drop=True)
