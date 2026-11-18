"""Candidate-level and observation-level evaluation of a scoring model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

THRESHOLD = 0.5


def pair_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Binary candidate classification metrics (threshold 0.5)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, float), 1e-15, 1 - 1e-15)
    y_pred = (proba >= THRESHOLD).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_true, proba) if both else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def top1_from_pairs(pairs: pd.DataFrame, proba: np.ndarray, obs: pd.DataFrame) -> dict:
    """Observation-level Top-1 accuracy and candidate miss rate.

    The denominator is every labelled observation in ``obs`` (including those
    with no candidates or whose label is not among the candidates — they count
    as wrong), so Top-1 is not inflated by candidate-generation failures.
    """
    df = pairs[["trip_id", "obs", "label"]].copy()
    df["p"] = proba
    best = df.sort_values("p", ascending=False, kind="stable").drop_duplicates(["trip_id", "obs"])
    labelled = obs[obs["label_arc"] >= 0]
    correct = labelled.merge(best[["trip_id", "obs", "label"]], on=["trip_id", "obs"], how="left")["label"]
    n = len(labelled)
    return {
        "top1_accuracy": float(correct.fillna(0).sum() / n) if n else float("nan"),
        "candidate_miss_rate": float(1 - labelled["label_in_candidates"].mean()) if n else float("nan"),
        "n_observations": int(n),
    }


def evaluate(pairs: pd.DataFrame, proba: np.ndarray, obs: pd.DataFrame) -> dict:
    out = pair_metrics(pairs["label"].to_numpy(), proba)
    out.update(top1_from_pairs(pairs, proba, obs))
    return out
