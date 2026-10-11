"""Stage 2 - Data preprocessing.
Median imputation of missing values followed by standardisation. The preprocessor is always
fitted on the training data only (inside each cross-validation fold) to avoid data leakage."""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def missing_report(X, A=None):
    import pandas as pd
    out = pd.DataFrame({"Missing (%)": (100 * X.isna().mean()).round(1)})
    if A is not None:
        for g in sorted(A.unique()):
            out[f"Missing (%) [{g}]"] = (100 * X[(A == g).values].isna().mean()).round(1)
    return out.sort_values("Missing (%)", ascending=False)
