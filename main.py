"""
FairCKD command-line runner
===========================
Run every experiment (tables, figures, Excel + PDF reports, trained models):

    python main.py --all

Run one configuration:

    python main.py --dataset uci    --attribute "Age group"      --features screening
    python main.py --dataset nhanes --attribute "Race/ethnicity" --features routine --repeats 5

List the available options:

    python main.py --list
"""
import argparse

from fairckd.data import load
from fairckd.models import MODEL_NAMES
from fairckd.pipeline import run_experiment

EXPERIMENTS = [
    # dataset, feature-set keyword, attribute, mitigation models, under-represented group
    ("uci", "screening", "Age group", ["Logistic Regression", "Random Forest", "XGBoost"], "Non-elderly (<60)"),
    ("uci", "full", "Age group", ["Logistic Regression", "XGBoost"], None),
    ("nhanes", "routine", "Sex", ["Logistic Regression", "XGBoost"], None),
    ("nhanes", "routine", "Race/ethnicity", ["Logistic Regression", "XGBoost"], None),
    ("nhanes", "routine", "Age group", ["Logistic Regression", "XGBoost"], None),
    ("nhanes", "routine", "Age × Sex", ["Logistic Regression", "XGBoost"], None),
    ("nhanes", "routine", "Age × Race", ["Logistic Regression", "XGBoost"], None),
]


def external_spec(ds):
    """NHANES 2015-16 as an external (temporal) validation cohort, if its files are present."""
    from fairckd.data import DATA_DIR, load_nhanes
    if ds != "nhanes" or not (DATA_DIR / "nhanes_2015").exists():
        return None
    spec = load_nhanes(csv=DATA_DIR / "nhanes_ckd_2015.csv", folder=DATA_DIR / "nhanes_2015", cycle="I")
    spec.name = "NHANES 2015-2016 (external)"
    return spec


def pick(spec, word):
    hits = [f for f in spec.feature_sets if word.lower() in f.lower()]
    if not hits:
        raise SystemExit(f"Feature set '{word}' not found. Options: {list(spec.feature_sets)}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description="FairCKD - fairness-aware CKD diagnosis pipeline")
    ap.add_argument("--all", action="store_true", help="run all experiments")
    ap.add_argument("--dataset", choices=["uci", "nhanes"])
    ap.add_argument("--attribute")
    ap.add_argument("--features", default="")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k in ["uci", "nhanes"]:
            s = load(k)
            print(f"\n{k}: {s.name}\n  feature sets: {list(s.feature_sets)}\n  attributes:   {list(s.sensitive)}")
        return
    if a.all:
        for ds, fw, attr, mm, rep in EXPERIMENTS:
            spec = load(ds)
            run_experiment(spec, pick(spec, fw), attr, mitigation_models=mm, representation_group=rep,
                           repeats_base=10 if ds == "uci" else a.repeats, repeats_mit=a.repeats,
                           external=external_spec(ds))
        return
    if not (a.dataset and a.attribute):
        ap.error("use --all, --list, or --dataset and --attribute")
    spec = load(a.dataset)
    fs = pick(spec, a.features) if a.features else list(spec.feature_sets)[0]
    run_experiment(spec, fs, a.attribute, repeats_base=a.repeats, repeats_mit=a.repeats)


if __name__ == "__main__":
    main()
