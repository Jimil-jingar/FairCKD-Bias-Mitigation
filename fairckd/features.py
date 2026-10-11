"""Stage 3 - Feature selection: rank features by mutual information with CKD status."""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from . import SEED


def feature_ranking(X: pd.DataFrame, y, labels: dict | None = None) -> pd.DataFrame:
    labels = labels or {}
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_classif(Xi, y, random_state=SEED)
    return (pd.DataFrame({"Feature": X.columns, "Description": [labels.get(c, c) for c in X.columns],
                          "Mutual information": mi.round(4),
                          "Missing (%)": (100 * X.isna().mean()).round(1).values})
            .sort_values("Mutual information", ascending=False).reset_index(drop=True))
