"""
FairCKD - Fairness-aware Machine Learning System for Automated Chronic Kidney Disease Diagnosis
===============================================================================================
Built by Jimil Jingar, Dept. of AI & DS, PIET, Parul University

Package layout (one module per stage of the proposed nine-stage framework):

    data.py           Stage 1  Clinical data acquisition (UCI CKD, NHANES 2017-18, custom CSV)
    preprocessing.py  Stage 2  Imputation + scaling (fitted on training folds only)
    features.py       Stage 3  Feature selection / ranking
    models.py         Stage 4  ML models (LR, DT, RF, SVM, KNN, NB, XGBoost)
    fairness.py       Stage 5-6 Bias detection + fairness metrics
    mitigation.py     Stage 7  Bias reduction (pre / in / post-processing)
    explain.py        Stage 8  Explainable AI (SHAP)
    diagnosis.py      Stage 9  Clinical decision support (trained fair model + explanation)
    evaluation.py     Repeated stratified cross-validation protocol
    pipeline.py       End-to-end experiment runner
    plots.py, report.py  Figures, Excel and PDF reports
"""
__version__ = "1.0.0"
SEED = 42

import warnings as _w
_w.filterwarnings("ignore", category=FutureWarning)
_w.filterwarnings("ignore", category=UserWarning)
