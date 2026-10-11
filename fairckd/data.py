"""Stage 1 - Clinical data acquisition.

Three data sources share one interface (`DatasetSpec.get`):
  * UCI Chronic Kidney Disease dataset (400 patients, India)   sensitive attribute: age group
  * NHANES 2017-2018 (US CDC survey, ~5,000 adults)              sensitive attributes: sex, race/ethnicity, age group
  * any custom CSV uploaded by the user
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AGE_CUTOFF = 60


@dataclass
class DatasetSpec:
    key: str
    name: str
    description: str
    df: pd.DataFrame                       # features + target + sensitive columns
    feature_sets: dict                     # name -> list of feature columns
    sensitive: dict                        # attribute name -> column in df
    proxy_cols: dict                       # attribute name -> model inputs that directly encode it
    labels: dict = field(default_factory=dict)       # column -> readable label
    binary: list = field(default_factory=list)       # 0/1 feature columns
    threshold: object = 0.5                # decision threshold, or "prevalence" for imbalanced data
    target: str = "ckd"

    def get(self, feature_set: str, attribute: str):
        """Return X (features), y (1 = CKD), A (sensitive group labels)."""
        cols = self.feature_sets[feature_set]
        scol = self.sensitive[attribute]
        d = self.df[self.df[scol].notna()].reset_index(drop=True)
        return d[cols].astype(float), d[self.target].astype(int), d[scol].astype(str).rename(attribute)

    def weights(self, attribute):
        """Survey weights aligned with get() rows (None if the dataset has none)."""
        if "weight" not in self.df:
            return None
        d = self.df[self.df[self.sensitive[attribute]].notna()].reset_index(drop=True)
        return d["weight"].astype(float)

    def label(self, c):
        return self.labels.get(c, c)


def _age_group(age):
    return np.where(age >= AGE_CUTOFF, f"Elderly (>={AGE_CUTOFF})", f"Non-elderly (<{AGE_CUTOFF})")


# --------------------------------------------------------------------------- UCI
UCI_NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
UCI_BINARY = {
    "rbc": {"normal": 0, "abnormal": 1}, "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1}, "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1}, "dm": {"no": 0, "yes": 1}, "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1}, "pe": {"no": 0, "yes": 1}, "ane": {"no": 0, "yes": 1},
}
UCI_LABELS = {
    "age": "Age (years)", "bp": "Blood pressure (mm Hg)", "sg": "Specific gravity",
    "al": "Albumin (0-5)", "su": "Sugar (0-5)", "rbc": "Red blood cells abnormal",
    "pc": "Pus cell abnormal", "pcc": "Pus cell clumps", "ba": "Bacteria",
    "bgr": "Blood glucose random (mg/dl)", "bu": "Blood urea (mg/dl)",
    "sc": "Serum creatinine (mg/dl)", "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)",
    "hemo": "Haemoglobin (g/dl)", "pcv": "Packed cell volume (%)",
    "wc": "WBC count (cells/cumm)", "rc": "RBC count (millions/cmm)",
    "htn": "Hypertension", "dm": "Diabetes mellitus", "cad": "Coronary artery disease",
    "appet": "Poor appetite", "pe": "Pedal oedema", "ane": "Anaemia",
}
UCI_FULL = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]
UCI_SCREENING = ["age", "bp", "bgr", "su", "htn", "dm", "cad", "appet", "pe", "ane"]


def load_uci(path=DATA_DIR / "uci_ckd.csv") -> DatasetSpec:
    raw = pd.read_csv(path)
    df = pd.DataFrame(index=raw.index)
    for c in UCI_NUMERIC:                       # stray '\t?' etc. -> NaN
        df[c] = pd.to_numeric(raw[c].astype(str).str.strip(), errors="coerce")
    for c, m in UCI_BINARY.items():
        df[c] = raw[c].astype(str).str.strip().map(m)
    df["ckd"] = raw["classification"].astype(str).str.strip().map({"ckd": 1, "notckd": 0})
    df = df[df["age"].notna() & df["ckd"].notna()].reset_index(drop=True)   # age = sensitive attr.
    df["age_group"] = _age_group(df["age"])
    return DatasetSpec(
        key="uci", name="UCI Chronic Kidney Disease",
        description="400 patients (Apollo Hospitals, Tamil Nadu, India), 24 clinical attributes; "
                    "391 with known age are used. Sensitive attribute: age group (>=60 vs <60).",
        df=df[UCI_FULL + ["ckd", "age_group"]],
        feature_sets={"Full clinical panel (24 features)": UCI_FULL,
                      "Primary-care screening (10 features, no kidney labs)": UCI_SCREENING},
        sensitive={"Age group": "age_group"},
        proxy_cols={"Age group": ["age"]},
        labels=UCI_LABELS, binary=list(UCI_BINARY), threshold=0.5,
    )


# --------------------------------------------------------------------------- NHANES
NHANES_FILES = ["DEMO_J", "BIOPRO_J", "ALB_CR_J", "CBC_J", "BPX_J", "BMX_J", "DIQ_J"]
NHANES_RACE = {1: "Hispanic", 2: "Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
               6: "Non-Hispanic Asian", 7: "Other/Multi-racial"}
NHANES_LABELS = {
    "age": "Age (years)", "female": "Female", "sbp": "Systolic BP (mm Hg)", "dbp": "Diastolic BP (mm Hg)",
    "htn": "Hypertension (BP >= 140/90)", "dm": "Diabetes (self-reported)", "bmi": "BMI (kg/m2)",
    "glucose": "Serum glucose (mg/dl)", "bun": "Blood urea nitrogen (mg/dl)",
    "albumin_serum": "Serum albumin (g/dl)", "uric_acid": "Uric acid (mg/dl)",
    "sodium": "Sodium (mmol/L)", "potassium": "Potassium (mmol/L)", "hemo": "Haemoglobin (g/dl)",
    "wbc": "WBC (1000 cells/uL)", "rbc": "RBC (million cells/uL)", "hct": "Haematocrit (%)",
}
NHANES_ROUTINE = ["age", "female", "sbp", "dbp", "htn", "dm", "bmi", "glucose", "bun", "albumin_serum",
                  "uric_acid", "sodium", "potassium", "hemo", "wbc", "rbc", "hct"]
NHANES_SCREENING = ["age", "female", "sbp", "dbp", "htn", "dm", "bmi", "glucose"]


def egfr_ckdepi_2021(scr, age, female):
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021)."""
    k = np.where(female, 0.7, 0.9); a = np.where(female, -0.241, -0.302); r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1)


def egfr_ckdepi_2009(scr, age, female, black):
    """Race-based CKD-EPI 2009 equation (Levey et al., Ann Intern Med 2009) - for label-bias analysis."""
    k = np.where(female, 0.7, 0.9); a = np.where(female, -0.329, -0.411); r = scr / k
    return (141 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.209 * 0.993 ** age
            * np.where(female, 1.018, 1) * np.where(black, 1.159, 1))


def build_nhanes(folder=DATA_DIR / "nhanes", cycle="J") -> pd.DataFrame:
    """Merge the 7 NHANES 2017-18 .XPT files into one analysis table (adults >= 18).
    CKD (KDIGO): eGFR < 60 or urine ACR >= 30 mg/g. Creatinine/ACR/eGFR define the label and are
    therefore never used as model inputs."""
    folder = Path(folder)

    def read(n):
        n = n[:-1] + cycle                                     # e.g. DEMO_J -> DEMO_I for 2015-16
        hit = [p for p in folder.iterdir() if p.stem.upper().startswith(n) and p.suffix.lower() == ".xpt"]
        if not hit:
            raise FileNotFoundError(f"{n}.xpt missing in {folder}")
        return pd.read_sas(sorted(hit)[0], format="xport")

    d = read("DEMO_J")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "WTMEC2YR"]]
    for n in NHANES_FILES[1:]:
        d = d.merge(read(n), on="SEQN", how="left")
    o = pd.DataFrame({"SEQN": d.SEQN.astype(int), "age": d.RIDAGEYR, "female": (d.RIAGENDR == 2).astype(int)})
    o["sex"] = np.where(o.female == 1, "Female", "Male")
    o["weight"] = d.WTMEC2YR                                   # NHANES MEC exam survey weight
    o["race"] = d.RIDRETH3.map(NHANES_RACE)
    sy = [c for c in d.columns if c.startswith("BPXSY")]; di = [c for c in d.columns if c.startswith("BPXDI")]
    o["sbp"] = d[sy].where(d[sy] >= 40).mean(axis=1)            # 0 = not measurable -> missing
    o["dbp"] = d[di].where(d[di] >= 20).mean(axis=1)
    o["htn"] = ((o.sbp >= 140) | (o.dbp >= 90)).astype(float).where(o.sbp.notna())
    o["dm"] = d.DIQ010.map({1: 1, 2: 0, 3: 0})
    o["bmi"] = d.BMXBMI
    for new, old in [("glucose", "LBXSGL"), ("bun", "LBXSBU"), ("albumin_serum", "LBXSAL"),
                     ("uric_acid", "LBXSUA"), ("sodium", "LBXSNASI"), ("potassium", "LBXSKSI"),
                     ("hemo", "LBXHGB"), ("wbc", "LBXWBCSI"), ("rbc", "LBXRBCSI"), ("hct", "LBXHCT"),
                     ("creatinine", "LBXSCR"), ("acr", "URDACT")]:
        o[new] = d[old]
    o = o[(o.age >= 18) & o.creatinine.notna() & o.acr.notna()].copy()
    fem, blk = o.female.values == 1, (o.race == "Non-Hispanic Black").values
    o["egfr_2021"] = egfr_ckdepi_2021(o.creatinine.values, o.age.values, fem)
    o["egfr_2009"] = egfr_ckdepi_2009(o.creatinine.values, o.age.values, fem, blk)
    o["ckd"] = ((o.egfr_2021 < 60) | (o.acr >= 30)).astype(int)
    o["ckd_2009"] = ((o.egfr_2009 < 60) | (o.acr >= 30)).astype(int)
    o["age_group"] = _age_group(o.age)
    o["age_short"] = np.where(o.age >= AGE_CUTOFF, "60+", "<60")
    o["age_sex"] = o.age_short + " " + o.sex                    # intersectional groups
    o["age_race"] = o.age_short + " " + o.race
    return o.reset_index(drop=True)


def load_nhanes(csv=DATA_DIR / "nhanes_ckd.csv", folder=DATA_DIR / "nhanes", cycle="J") -> DatasetSpec:
    csv = Path(csv)
    if not csv.exists() or "age_sex" not in pd.read_csv(csv, nrows=1).columns:
        build_nhanes(folder, cycle).to_csv(csv, index=False)
    df = pd.read_csv(csv)
    return DatasetSpec(
        key="nhanes", name="NHANES 2017-2018 (US CDC)",
        description=f"{len(df):,} US adults with blood and urine tests. CKD = eGFR < 60 (CKD-EPI 2021) or "
                    "urine ACR >= 30 mg/g. Creatinine, ACR and eGFR define the label and are not model inputs. "
                    "Sensitive attributes: sex, race/ethnicity, age group.",
        df=df,
        feature_sets={"Routine blood tests (17 features, no kidney-function tests)": NHANES_ROUTINE,
                      "Primary-care screening (8 features)": NHANES_SCREENING},
        sensitive={"Sex": "sex", "Race/ethnicity": "race", "Age group": "age_group",
                   "Age × Sex": "age_sex", "Age × Race": "age_race"},
        proxy_cols={"Sex": ["female"], "Race/ethnicity": [], "Age group": ["age"],
                    "Age × Sex": ["age", "female"], "Age × Race": ["age"]},
        labels=NHANES_LABELS, binary=["female", "htn", "dm"], threshold="prevalence",
    )


# --------------------------------------------------------------------------- custom CSV
def load_custom(df: pd.DataFrame, target: str, positive, sensitive: str, name="Custom dataset") -> DatasetSpec:
    """Generic loader: any CSV with a binary CKD column and a categorical sensitive column."""
    df = df.copy()
    y = (df[target].astype(str).str.strip().str.lower() == str(positive).strip().lower()).astype(int)
    A = df[sensitive].astype(str).str.strip()
    X = df.drop(columns=[target, sensitive])
    feats, out, binary = [], pd.DataFrame(index=df.index), []
    for c in X.columns:
        s = X[c]
        num = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
        if num.notna().mean() >= 0.8 * s.notna().mean() and s.notna().any():
            out[c] = num
        else:
            vals = s.astype(str).str.strip().str.lower().replace({"nan": np.nan})
            uniq = vals.dropna().unique()
            if len(uniq) == 2:
                pos = sorted(uniq)[-1] if not {"yes", "no"} <= set(uniq) else "yes"
                out[c] = (vals == pos).astype(float).where(vals.notna()); binary.append(c)
            elif 2 < len(uniq) <= 10:
                for u in uniq:
                    out[f"{c}={u}"] = (vals == u).astype(float).where(vals.notna()); binary.append(f"{c}={u}")
            else:
                continue
    feats = list(out.columns)
    out["ckd"], out["_group"] = y, A
    prev = y.mean()
    return DatasetSpec(key="custom", name=name,
                       description=f"{len(df)} records, {len(feats)} usable features, sensitive attribute '{sensitive}'.",
                       df=out, feature_sets={"All features": feats}, sensitive={sensitive: "_group"},
                       proxy_cols={sensitive: []}, binary=binary,
                       threshold="prevalence" if prev < 0.3 or prev > 0.7 else 0.5)


def load(key: str) -> DatasetSpec:
    return {"uci": load_uci, "nhanes": load_nhanes}[key]()


def group_summary(X, y, A) -> pd.DataFrame:
    rows = []
    for g in sorted(A.unique()):
        m = (A == g).values
        rows.append({"Group": g, "Patients": int(m.sum()), "Share (%)": round(100 * m.mean(), 1),
                     "CKD cases": int(y[m].sum()), "CKD prevalence (%)": round(100 * y[m].mean(), 1),
                     "Missing values / patient": round(X[m].isna().sum(axis=1).mean(), 2)})
    rows.append({"Group": "All", "Patients": len(y), "Share (%)": 100.0, "CKD cases": int(y.sum()),
                 "CKD prevalence (%)": round(100 * y.mean(), 1),
                 "Missing values / patient": round(X.isna().sum(axis=1).mean(), 2)})
    return pd.DataFrame(rows)
