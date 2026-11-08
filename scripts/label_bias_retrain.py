# Run from the repository root:  python scripts/label_bias_retrain.py
"""Label-bias retraining: same protocol as the baseline (stratified 5-fold, 5 repeats, prevalence threshold),
NHANES 2017-18, routine blood tests, race/ethnicity; only the CKD label changes (2021 race-free vs 2009 race-based)."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from fairckd.data import load
from fairckd.evaluation import cross_validate
spec = load("nhanes")
fs = [k for k in spec.feature_sets if "outine" in k][0]
X, y21, A = spec.get(fs, "Race/ethnicity")
d = spec.df[spec.df[spec.sensitive["Race/ethnicity"]].notna()].reset_index(drop=True)
y09 = d["ckd_2009"].astype(int)
assert (d[spec.target].astype(int).values == y21.values).all()
rows = []
for m in ["Logistic Regression", "XGBoost"]:
    for name, y in [("2021 race-free", y21), ("2009 race-based", y09)]:
        r = cross_validate(X, y, A, m, n_repeats=5, threshold=spec.threshold)
        rows.append({"Model": m, "CKD label": name, "CKD prevalence (%)": round(100 * y.mean(), 2),
                     "EOdds gap": f"{r['EOdds diff'].mean():.3f} ± {r['EOdds diff'].std(ddof=1):.3f}",
                     "Balanced acc.": f"{r['Balanced acc.'].mean():.3f} ± {r['Balanced acc.'].std(ddof=1):.3f}" if 'Balanced acc.' in r else ""})
        print(rows[-1], flush=True)
out = sys.argv[1] if len(sys.argv) > 1 else "results/nhanes__race-ethnicity__routine-blood-tests/tables/label_bias_retrain.csv"
pd.DataFrame(rows).to_csv(out, index=False)
