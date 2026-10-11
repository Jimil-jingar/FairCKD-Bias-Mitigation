"""Stage 4 - Machine learning model development."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import SEED

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes", "XGBoost"]


def make_classifier(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=SEED, n_jobs=1)
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=7)
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="logloss", random_state=SEED,
                             n_jobs=1, verbosity=0)
    raise ValueError(f"Unknown model {name}")


class Float64Proba(ClassifierMixin, BaseEstimator):
    """Wrapper used by the post-processing step.
    * returns float64 probabilities (XGBoost returns float32, which Fairlearn cannot handle with pandas >= 3)
    * optionally returns stored OUT-OF-FOLD scores for the training matrix, so that group thresholds
      are learned on honest (not over-fitted) scores."""

    def __init__(self, est=None, train_X=None, train_proba=None):
        self.est, self.train_X, self.train_proba = est, train_X, train_proba
        if est is not None and hasattr(est, "classes_"):
            self.classes_ = est.classes_

    def fit(self, X, y, **kw):
        self.est.fit(X, y, **kw)
        self.classes_ = self.est.classes_
        return self

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.train_proba is not None and X.shape == self.train_X.shape and np.array_equal(X, self.train_X):
            return np.asarray(self.train_proba, dtype=np.float64)
        return np.asarray(self.est.predict_proba(X), dtype=np.float64)
