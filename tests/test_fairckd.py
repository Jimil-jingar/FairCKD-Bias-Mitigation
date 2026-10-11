"""Automated tests for the FairCKD package.   Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fairckd import data as D
from fairckd.evaluation import cross_validate, downsample_group
from fairckd.fairness import fairness_report, group_rates, reference_values, reweighing_weights
from fairckd.mitigation import METHODS, fit_predict
from fairckd.models import MODEL_NAMES, make_classifier


# ---------------------------------------------------------------- data
def test_uci_loader():
    spec = D.load_uci()
    X, y, A = spec.get("Full clinical panel (24 features)", "Age group")
    assert X.shape == (391, 24)
    assert set(y.unique()) == {0, 1}
    assert set(A.unique()) == {"Elderly (>=60)", "Non-elderly (<60)"}
    assert y.sum() == 242                       # CKD cases after dropping missing age


def test_nhanes_loader():
    spec = D.load_nhanes()
    for attr in spec.sensitive:
        X, y, A = spec.get(list(spec.feature_sets)[0], attr)
        assert len(X) == len(y) == len(A) > 4000
    # label inputs must never be model features (no leakage)
    for fs in spec.feature_sets.values():
        assert not {"creatinine", "acr", "egfr_2021", "egfr_2009", "ckd", "ckd_2009"} & set(fs)


def test_egfr_equations():
    # CKD-EPI 2021, woman 50 y, creatinine 1.0 mg/dl  ->  ~68.6 mL/min/1.73m2
    assert D.egfr_ckdepi_2021(np.array([1.0]), np.array([50]), np.array([True]))[0] == pytest.approx(68.6, abs=0.5)
    # 2009 equation gives Black patients a 15.9 % higher eGFR for the same creatinine
    a = D.egfr_ckdepi_2009(np.array([1.2]), np.array([60]), np.array([False]), np.array([True]))[0]
    b = D.egfr_ckdepi_2009(np.array([1.2]), np.array([60]), np.array([False]), np.array([False]))[0]
    assert a / b == pytest.approx(1.159)


def test_custom_loader():
    df = pd.DataFrame({"age": [30, 40, 50, 60, 70, 80] * 5, "bp": [80, 90, 70, 100, 85, 95] * 5,
                       "dm": ["yes", "no"] * 15, "sex": ["M", "F", "M"] * 10, "class": ["ckd", "notckd", "ckd"] * 10})
    spec = D.load_custom(df, "class", "ckd", "sex")
    X, y, A = spec.get("All features", "sex")
    assert list(X.columns) == ["age", "bp", "dm"] and y.sum() == 20 and set(A) == {"M", "F"}


# ---------------------------------------------------------------- fairness metrics
def test_fairness_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    p = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    A = np.array(["a"] * 4 + ["b"] * 4)
    r = fairness_report(y, p, A)
    # group a: perfect; group b: TPR 0.5, FPR 0.5
    assert r["EOpp diff"] == pytest.approx(0.5)
    assert r["EOdds diff"] == pytest.approx(0.5)
    assert r["DP diff"] == pytest.approx(0.0)              # both groups flagged 50 %
    assert r["DI ratio"] == pytest.approx(1.0)
    assert r["Acc. gap"] == pytest.approx(0.5)
    g = group_rates(y, p, A)
    assert g.loc["b", "TPR"] == 0.5 and g.loc["a", "FPR"] == 0.0


def test_perfect_classifier_reference():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    A = np.array(["a"] * 4 + ["b"] * 4)
    ref = reference_values(y, A)
    assert ref["EOdds diff"] == 0 and ref["EOpp diff"] == 0
    assert ref["DP diff"] == pytest.approx(0.5)            # base rates 0.75 vs 0.25


def test_reweighing_makes_group_and_label_independent():
    rng = np.random.default_rng(0)
    A = rng.choice(["a", "b"], 500, p=[0.3, 0.7])
    y = (rng.random(500) < np.where(A == "a", 0.8, 0.3)).astype(int)
    w = reweighing_weights(y, A)
    for g in ["a", "b"]:
        m = A == g
        assert np.average(y[m], weights=w[m]) == pytest.approx(np.average(y, weights=w), abs=1e-9)


# ---------------------------------------------------------------- models + mitigation
@pytest.fixture(scope="module")
def uci_screen():
    return D.load_uci().get("Primary-care screening (10 features, no kidney labs)", "Age group")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_trains(name, uci_screen):
    X, y, A = uci_screen
    m = make_classifier(name).fit(X.fillna(X.median()).values, y)
    assert m.predict_proba(X.fillna(X.median()).values[:5]).shape == (5, 2)


@pytest.mark.parametrize("method", METHODS)
def test_every_mitigation_method(method, uci_screen):
    X, y, A = uci_screen
    tr, te = np.arange(0, 300), np.arange(300, len(y))
    pred, proba = fit_predict(method, "Logistic Regression", X.iloc[tr], y.iloc[tr], A.iloc[tr], X.iloc[te], A.iloc[te],
                              proxy_cols=["age"])
    assert len(pred) == len(te) and set(np.unique(pred)) <= {0, 1}
    assert (pred == y.iloc[te].values).mean() > 0.7


def test_cross_validation_reproducible(uci_screen):
    X, y, A = uci_screen
    a = cross_validate(X, y, A, "Logistic Regression", n_repeats=2)
    b = cross_validate(X, y, A, "Logistic Regression", n_repeats=2)
    pd.testing.assert_frame_equal(a, b)
    assert a.Accuracy.mean() > 0.85


def test_downsample_only_touches_one_group(uci_screen):
    X, y, A = uci_screen
    f = downsample_group("Non-elderly (<60)", 0.25)
    tr = f(np.arange(len(y)), y, A, np.random.default_rng(0))
    assert (A.iloc[tr] == "Elderly (>=60)").sum() == (A == "Elderly (>=60)").sum()
    assert (A.iloc[tr] == "Non-elderly (<60)").sum() < 0.3 * (A == "Non-elderly (<60)").sum()


# ---------------------------------------------------------------- decision support
def test_patient_prediction_and_pdf():
    from fairckd.diagnosis import predict_patient, train_final_model
    from fairckd.report import patient_report_pdf
    spec = D.load_uci()
    b = train_final_model(spec, "Full clinical panel (24 features)", "Age group", save=False)
    sick = {"age": 62, "bp": 90, "sg": 1.010, "al": 3, "su": 1, "rbc": 1, "pc": 1, "pcc": 1, "ba": 0, "bgr": 210,
            "bu": 90, "sc": 5.2, "sod": 131, "pot": 5.1, "hemo": 8.9, "pcv": 28, "wc": 11000, "rc": 3.2,
            "htn": 1, "dm": 1, "cad": 0, "appet": 1, "pe": 1, "ane": 1}
    healthy = {"age": 35, "bp": 70, "sg": 1.025, "al": 0, "su": 0, "rbc": 0, "pc": 0, "pcc": 0, "ba": 0, "bgr": 100,
               "bu": 25, "sc": 0.8, "sod": 142, "pot": 4.3, "hemo": 15.5, "pcv": 46, "wc": 7500, "rc": 5.3,
               "htn": 0, "dm": 0, "cad": 0, "appet": 0, "pe": 0, "ane": 0}
    p1, pos1, sv1 = predict_patient(b, sick)
    p2, pos2, _ = predict_patient(b, healthy)
    assert pos1 and p1 > 0.9 and not pos2 and p2 < 0.1
    assert len(sv1.values) == 24
    pdf = patient_report_pdf(b, sick, p1, pos1, patient_id="TEST-1", group="Elderly (>=60)")
    assert pdf[:4] == b"%PDF" and len(pdf) > 2000


# ---------------------------------------------------------------- upgrades
def test_weighted_metrics_equal_unweighted_with_unit_weights():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200); p = rng.integers(0, 2, 200); A = rng.choice(["a", "b"], 200)
    r1 = fairness_report(y, p, A); r2 = fairness_report(y, p, A, w=np.ones(200))
    for k in ["Accuracy", "Balanced acc.", "EOdds diff", "DP diff"]:
        assert r1[k] == pytest.approx(r2[k])


def test_survey_weights_change_estimates():
    y = np.array([1, 1, 0, 0]); p = np.array([1, 0, 0, 0]); A = np.array(["a"] * 4)
    assert fairness_report(y, p, A, w=np.array([3, 1, 1, 1]))["Recall"] == pytest.approx(0.75)


def test_calibration_by_group():
    from fairckd.fairness import calibration_by_group
    y = np.array([0, 0, 1, 1] * 25); pr = np.array([0.1, 0.1, 0.9, 0.9] * 25); A = np.array(["a", "b"] * 50)
    t, c = calibration_by_group(y, pr, A)
    assert set(t.Group) == {"a", "b"} and (t.ECE < 0.11).all()


def test_bootstrap_compare_detects_improvement():
    from fairckd.evaluation import bootstrap_compare
    rng = np.random.default_rng(0)
    A = np.array(["a"] * 200 + ["b"] * 200); y = rng.integers(0, 2, 400)
    biased = y.copy(); biased[(A == "b") & (y == 1)] = (rng.random(((A == "b") & (y == 1)).sum()) < 0.5)
    out = bootstrap_compare(y, A, biased, y, n_boot=200)
    assert out["EOdds diff"]["mean"] < 0 and out["EOdds diff"]["p"] < 0.05


def test_select_rule_respects_accuracy_tolerance():
    from fairckd.mitigation import BASELINE, THRESH, EXPGRAD
    from fairckd.select import select_rule
    sc = pd.DataFrame({"Balanced acc.": [0.90, 0.80, 0.885], "EOdds diff": [0.10, 0.01, 0.03]},
                      index=pd.MultiIndex.from_tuples([("LR", BASELINE), ("LR", EXPGRAD), ("LR", THRESH)]))
    assert select_rule(sc, 0.03, "fair") == ("LR", THRESH)        # EXPGRAD too inaccurate
    assert select_rule(sc, 0.03, "accuracy") == ("LR", BASELINE)


def test_pareto_front():
    from fairckd.select import pareto_front
    pts = pd.DataFrame({"EOdds diff": [0.1, 0.05, 0.08], "Balanced acc.": [0.9, 0.85, 0.84]})
    assert pareto_front(pts).tolist() == [True, True, False]


def test_hybrid_method_runs(uci_screen):
    from fairckd.mitigation import HYBRID
    X, y, A = uci_screen
    pred, _ = fit_predict(HYBRID, "XGBoost", X.iloc[:300], y.iloc[:300], A.iloc[:300], X.iloc[300:], A.iloc[300:])
    assert (pred == y.iloc[300:].values).mean() > 0.7


def test_intersectional_attributes():
    spec = D.load_nhanes()
    X, y, A = spec.get(list(spec.feature_sets)[0], "Age × Race")
    assert A.nunique() == 10 and spec.weights("Age × Race").notna().all()


def test_group_smote_balances_every_cell():
    from fairckd.mitigation import group_smote
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 3))
    y = np.r_[np.ones(8), np.zeros(37), np.ones(3), np.zeros(12)].astype(int)      # group a: 8/37, group b: 3/12
    A = np.r_[np.repeat("a", 45), np.repeat("b", 15)]
    Zs, ys = group_smote(Z, y, A)
    assert len(Zs) == len(ys) == 4 * 37 and ys.mean() == 0.5
    assert np.isfinite(Zs).all()


def test_reject_option_only_changes_the_band():
    from fairckd.mitigation import reject_option
    p = np.array([0.1, 0.45, 0.55, 0.9, 0.48, 0.52]); A = np.array(["lo", "lo", "lo", "hi", "hi", "hi"])
    pred = reject_option(p, A, 0.5, ["lo"], 0.06)
    assert list(pred) == [0, 1, 1, 1, 0, 0]


def test_lime_explanation(uci_screen):
    from fairckd.explain import lime_explain
    from fairckd.models import make_classifier
    from fairckd.preprocessing import make_preprocessor
    X, y, A = uci_screen
    pre = make_preprocessor().fit(X)
    m = make_classifier("Logistic Regression").fit(pre.transform(X), y)
    t, info = lime_explain(m, pre, X.iloc[[0]], n_samples=500)
    assert len(t) == X.shape[1] and t.Contribution.abs().is_monotonic_decreasing
    assert 0 <= info["model_prediction"] <= 1
