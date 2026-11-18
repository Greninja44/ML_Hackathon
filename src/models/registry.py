"""Candidate-scoring model zoo.

Task: binary classification of (observation, candidate arc) pairs — is this
candidate the reference arc? Every model must expose ``predict_proba`` because
the HMM consumes probabilities as emission scores.

Each entry records its algorithm family and the *initial* (untuned)
hyperparameters, which the report must list (Deliverables §3.2 B).
Scaling is placed inside each pipeline, so it is re-fitted on the training
folds only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    family: str
    build: Callable[[dict, int], Pipeline]
    params: dict
    loss: str
    max_train_rows: int | None = None  # subsample for models that scale badly (KNN)
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _logreg(p, seed):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(**p, random_state=seed))])


def _gnb(p, seed):
    return Pipeline([("scale", StandardScaler()), ("clf", GaussianNB(**p))])


def _knn(p, seed):
    return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(**p))])


def _tree(p, seed):
    return Pipeline([("clf", DecisionTreeClassifier(**p, random_state=seed))])


def _rf(p, seed):
    return Pipeline([("clf", RandomForestClassifier(**p, random_state=seed))])


def _et(p, seed):
    return Pipeline([("clf", ExtraTreesClassifier(**p, random_state=seed))])


def _hgb(p, seed):
    return Pipeline([("clf", HistGradientBoostingClassifier(**p, random_state=seed))])


def _mlp(p, seed):
    return Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(**p, random_state=seed))])


def default_specs(n_jobs: int = -1) -> dict[str, ModelSpec]:
    specs = [
        ModelSpec("logistic_regression", "Linear", _logreg,
                  {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}, "binary cross-entropy + L2"),
        ModelSpec("gaussian_nb", "Probabilistic", _gnb, {"var_smoothing": 1e-9},
                  "maximum likelihood (class-conditional Gaussians)"),
        ModelSpec("knn", "Instance-based", _knn,
                  {"n_neighbors": 25, "weights": "distance", "n_jobs": n_jobs},
                  "none (lazy learner); probability = weighted neighbour vote",
                  max_train_rows=150_000, notes="trained on a random 150k-row subsample for tractable inference"),
        ModelSpec("decision_tree", "Tree-based", _tree,
                  {"max_depth": 12, "min_samples_leaf": 20}, "Gini impurity"),
        ModelSpec("random_forest", "Ensemble (bagging)", _rf,
                  {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": n_jobs},
                  "Gini impurity, bootstrap aggregation"),
        ModelSpec("extra_trees", "Ensemble (randomised trees)", _et,
                  {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": n_jobs},
                  "Gini impurity, random split thresholds"),
        ModelSpec("hist_gradient_boosting", "Boosting", _hgb,
                  {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.0,
                   "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20},
                  "binary log loss (gradient boosting)"),
        ModelSpec("mlp", "Neural network", _mlp,
                  {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3,
                   "max_iter": 200, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10},
                  "binary cross-entropy (Adam)"),
    ]
    out = {s.name: s for s in specs}
    out.update(_optional_specs(n_jobs))
    return out


def _optional_specs(n_jobs: int) -> dict[str, ModelSpec]:
    """XGBoost / LightGBM need the OpenMP runtime; include them only if they import."""
    out = {}
    try:
        from xgboost import XGBClassifier  # noqa: F401

        def _xgb(p, seed):
            return Pipeline([("clf", XGBClassifier(**p, random_state=seed))])

        out["xgboost"] = ModelSpec("xgboost", "Boosting", _xgb,
                                   {"n_estimators": 400, "learning_rate": 0.1, "max_depth": 6,
                                    "subsample": 0.9, "colsample_bytree": 0.9, "n_jobs": n_jobs,
                                    "eval_metric": "logloss"}, "binary log loss")
    except Exception as exc:  # ImportError or XGBoostError (missing libomp)
        log.info("XGBoost unavailable (%s); skipping", type(exc).__name__)
    try:
        from lightgbm import LGBMClassifier  # noqa: F401

        def _lgbm(p, seed):
            return Pipeline([("clf", LGBMClassifier(**p, random_state=seed))])

        out["lightgbm"] = ModelSpec("lightgbm", "Boosting", _lgbm,
                                    {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                                     "n_jobs": n_jobs, "verbose": -1}, "binary log loss")
    except Exception as exc:
        log.info("LightGBM unavailable (%s); skipping", type(exc).__name__)
    return out
