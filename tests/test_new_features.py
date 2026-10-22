"""Tests for the features added in FairCKD 1.2: income attribute, adversarial debiasing, feature-selection
methods, proxy audit, drift monitoring, PDP/ICE and probability-scale explanations."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fairckd import data as D
from fairckd.adversarial import AdversarialDebiasing, fit_tuned
from fairckd.drift import data_drift, fairness_drift, prevalence_drift, psi
from fairckd.features import feature_ranking
from fairckd.mitigation import ADVERSARIAL, EXPGRAD, fit_predict, supports
from fairckd.proxy import group_predictability, proxy_audit


@pytest.fixture(scope="module")
def uci():
    spec = D.load_uci()
    return spec, *spec.get("Primary-care screening (10 features, no kidney labs)", "Age group")


def test_income_attribute():
    spec = D.load_nhanes()
    X, y, A = spec.get(list(spec.feature_sets)[0], "Income group")
    assert set(A) == {D.LOW_INCOME, D.HIGHER_INCOME}
    assert 4000 < len(y) < 5063                   # patients with unknown income are excluded for this attribute
    assert "pir" not in X.columns and "income_group" not in X.columns


def test_supports_rules():
    assert not supports(EXPGRAD, "KNN") and supports(EXPGRAD, "XGBoost")
    assert supports(ADVERSARIAL, "Logistic Regression") and not supports(ADVERSARIAL, "XGBoost")


def test_adversarial_zero_alpha_matches_logistic(uci):
    _, X, y, A = uci
    Z = X.fillna(X.median()).values
    Z = (Z - Z.mean(0)) / Z.std(0)
    m = AdversarialDebiasing(alpha=0.0, epochs=5).fit(Z, y.values, A.values)
    p = m.predict_proba(Z)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(1), 1)
    assert ((p[:, 1] >= 0.5) == y.values).mean() > 0.8


def test_adversarial_reduces_group_signal():
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.integers(0, 2, n)
    x1 = rng.normal(size=n) + 1.5 * a                # strongly group-linked feature
    x2 = rng.normal(size=n)
    y = (x1 + x2 + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    Z = np.column_stack([x1, x2])
    A = np.where(a == 1, "g1", "g0")
    base = AdversarialDebiasing(alpha=0.0).fit(Z, y, A).predict_proba(Z)[:, 1]
    adv = AdversarialDebiasing(alpha=3.0).fit(Z, y, A).predict_proba(Z)[:, 1]
    gap = lambda p: abs(p[a == 1].mean() - p[a == 0].mean())
    assert gap(adv) < gap(base)


def test_adversarial_tuning_returns_grid_value(uci):
    _, X, y, A = uci
    Z = X.fillna(X.median()).values
    Z = (Z - Z.mean(0)) / Z.std(0)
    m, alpha = fit_tuned(Z, y.values, A.values, cut=lambda p, yt: (p >= 0.5).astype(int))
    assert alpha in (0.5, 1.0, 3.0, 10.0) and len(m.alpha_scores_) == 4


def test_adversarial_in_pipeline(uci):
    _, X, y, A = uci
    tr, te = np.arange(0, 300), np.arange(300, len(y))
    pred, proba = fit_predict(ADVERSARIAL, "Logistic Regression", X.iloc[tr], y.iloc[tr], A.iloc[tr],
                              X.iloc[te], A.iloc[te])
    assert len(pred) == len(te) and proba is not None
    with pytest.raises(ValueError):
        fit_predict(ADVERSARIAL, "XGBoost", X.iloc[tr], y.iloc[tr], A.iloc[tr], X.iloc[te], A.iloc[te])


def test_feature_ranking_methods(uci):
    spec, X, y, A = uci
    r = feature_ranking(X, y, spec.labels)
    assert {"Mutual information", "|Correlation| with CKD", "RFE rank", "Consensus rank"} <= set(r.columns)
    assert sorted(r["RFE rank"]) == list(range(1, X.shape[1] + 1))
    assert r["|Correlation| with CKD"].between(0, 1).all()


def test_proxy_audit_detects_age(uci):
    spec, X, y, A = uci
    feats, auc = proxy_audit(X, A, spec.labels, ["age"])
    assert len(auc) == 2                                   # with and without the direct proxy
    assert auc.iloc[0]["AUC (Logistic Regression)"] > 0.95  # age itself encodes age group
    assert feats.iloc[0].Feature == "age"


def test_group_predictability_random_is_chance():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=list("abc"))
    A = rng.choice(["x", "y"], 600)
    auc = group_predictability(X, A).iloc[0]
    assert 0.4 < auc["AUC (Logistic Regression)"] < 0.6


def test_psi():
    rng = np.random.default_rng(2)
    a = rng.normal(size=4000)
    assert psi(a, rng.normal(size=4000)) < 0.05
    assert psi(a, rng.normal(1.0, 1, size=4000)) > 0.2
    assert psi(np.r_[np.zeros(500), np.ones(500)], np.r_[np.zeros(500), np.ones(500)]) == pytest.approx(0, abs=1e-9)


def test_drift_tables():
    X = pd.DataFrame({"f": np.r_[np.zeros(100), np.ones(100)]})
    Xn = pd.DataFrame({"f": np.r_[np.zeros(50), np.ones(150)]})
    A = np.array(["a", "b"] * 100)
    dd = data_drift(X, A, Xn, A, {"f": "Feature F"})
    assert dd.loc[0, "Feature"] == "Feature F" and dd.loc[0, "PSI (all)"] > 0.1
    pv = prevalence_drift(np.r_[np.ones(20), np.zeros(180)], A, np.r_[np.ones(40), np.zeros(160)], A)
    assert pv["Change (pp)"].sum() > 0
    raw = pd.DataFrame({"Model": ["LR"] * 2, "Method": ["Baseline"] * 2, "Balanced acc.": [0.7, 0.72],
                        "EOdds diff": [0.1, 0.12]})
    ext = pd.DataFrame({"Model": ["LR"], "Method": ["Baseline"], "Balanced acc.": [0.69], "EOdds diff": [0.3]})
    fd = fairness_drift(raw, ext, {})
    assert fd.loc[0, "Δ EOdds gap"] == pytest.approx(0.19) and fd.loc[0, "Fairness held?"].startswith("No")


def test_pdp_ice_and_probability_contributions(uci):
    from fairckd.explain import explain, pdp_ice, probability_contributions
    from fairckd.models import make_classifier
    from fairckd.preprocessing import make_preprocessor
    spec, X, y, A = uci
    pre = make_preprocessor().fit(X)
    m = make_classifier("XGBoost").fit(pre.transform(X), y)
    d = pdp_ice(m, pre, X, A, ["age", "bgr"], spec.labels, grid_points=8, ice_patients=5)
    assert set(d.Kind) == {"PDP", "Group PDP", "ICE"} and d["CKD probability"].between(0, 1).all()
    sv = explain(m, pre, X.iloc[:3], spec.labels)
    ct, p0, p1 = probability_contributions(sv[0], top=5)
    prob = m.predict_proba(pre.transform(X.iloc[[0]]))[0, 1]
    assert p1 == pytest.approx(prob, abs=1e-4)
    assert ct["Contribution (pp)"].sum() == pytest.approx(100 * (p1 - p0), abs=1e-6)


def test_reference_patients(uci):
    from fairckd.diagnosis import predict_patient, train_final_model
    spec, X, y, A = uci
    b = train_final_model(spec, "Full clinical panel (24 features)", "Age group", save=False)
    p_h, _, _ = predict_patient(b, b["examples"]["healthy"])
    p_c, _, _ = predict_patient(b, b["examples"]["ckd"])
    assert p_h < 0.5 < p_c
