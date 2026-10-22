"""Stage 9 - Clinical decision support.
Trains the final fairness-aware model (XGBoost + Reweighing) on all data of a dataset/feature set,
saves it, and predicts + explains individual patients."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import ROOT
from .explain import explain, lime_explain
from .fairness import reweighing_weights
from .models import make_classifier
from .preprocessing import make_preprocessor

MODEL_DIR = ROOT / "models"


def model_path(spec, feature_set, attribute):
    fs = "".join(ch for ch in feature_set.split("(")[0].strip().lower().replace(" ", "_") if ch.isalnum() or ch == "_")
    at = attribute.lower().replace("×", "x").replace("/", "_").replace(" ", "_")
    return MODEL_DIR / f"{spec.key}__{fs}__{at}.joblib"


def train_final_model(spec, feature_set, attribute, model_name="XGBoost", save=True):
    X, y, A = spec.get(feature_set, attribute)
    pre = make_preprocessor().fit(X)
    model = make_classifier(model_name)
    model.fit(pre.transform(X), y, sample_weight=reweighing_weights(y, A))
    thr = float(y.mean()) if spec.threshold == "prevalence" else float(spec.threshold)
    bundle = {"dataset": spec.name, "dataset_key": spec.key, "feature_set": feature_set,
              "attribute": attribute, "model_name": f"{model_name} + Reweighing", "features": list(X.columns),
              "preprocessor": pre, "model": model, "threshold": thr, "labels": spec.labels,
              "binary": [c for c in spec.binary if c in X.columns],
              "medians": X.median().to_dict(), "mins": X.min().to_dict(), "maxs": X.max().to_dict(),
              "n_train": len(y), "prevalence": float(y.mean()),
              "examples": reference_patients(X, y)}
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, model_path(spec, feature_set, attribute))
    return bundle


def reference_patients(X, y) -> dict:
    """Two ready-made demo patients: the median non-CKD patient and the median CKD patient of the dataset.
    (The overall median is not a neutral default: in a CKD-enriched dataset like UCI it is a CKD patient.)"""
    y = np.asarray(y)
    out = {}
    for name, lab in [("healthy", 0), ("ckd", 1)]:
        sub = X[y == lab]
        med = sub.median()
        for c in X.columns:                          # binary features: most common value, not the median
            if X[c].dropna().isin([0, 1]).all():
                med[c] = float(sub[c].mode().iloc[0]) if sub[c].notna().any() else 0.0
        out[name] = med.to_dict()
    return out


def load_model(path):
    return joblib.load(path)


def predict_patient(bundle, values: dict):
    """values: feature -> value (np.nan if unknown). Returns probability, decision, SHAP explanation."""
    x = pd.DataFrame([{f: values.get(f, np.nan) for f in bundle["features"]}], dtype=float)
    p = float(bundle["model"].predict_proba(bundle["preprocessor"].transform(x))[0, 1])
    sv = explain(bundle["model"], bundle["preprocessor"], x, bundle["labels"])
    return p, p >= bundle["threshold"], sv[0]


def lime_patient(bundle, values: dict, n_samples: int = 3000):
    """LIME explanation of one patient for a saved model bundle."""
    x = pd.DataFrame([{f: values.get(f, np.nan) for f in bundle["features"]}], dtype=float)
    return lime_explain(bundle["model"], bundle["preprocessor"], x, bundle["labels"], bundle.get("binary", ()),
                        n_samples=n_samples)
