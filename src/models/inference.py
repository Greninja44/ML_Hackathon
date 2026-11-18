"""Stable inference interface between the ML model and the route reconstructor.

Contract
--------
``CandidateScorer.predict_candidates(features: pd.DataFrame) -> np.ndarray``

* input: the DataFrame produced by ``src.data.features.build_features`` (one row
  per (observation, candidate arc) pair; must contain ``FEATURE_COLUMNS``);
* output: float array of the same length, ``P(candidate is the reference arc)``.

The artifact directory contains:

* ``model.joblib``   – a fitted sklearn ``Pipeline`` (preprocessing + estimator)
* ``metadata.json``  – model name, feature list, params, training info
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ModelArtifactError(RuntimeError):
    pass


class CandidateScorer:
    def __init__(self, pipeline, feature_columns: list[str], metadata: dict | None = None):
        self.pipeline = pipeline
        self.feature_columns = list(feature_columns)
        self.metadata = metadata or {}

    @property
    def name(self) -> str:
        return self.metadata.get("model_name", type(self.pipeline).__name__)

    def predict_candidates(self, features: pd.DataFrame) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0)
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing}")
        X = features[self.feature_columns].to_numpy(dtype=float)
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, d / "model.joblib", compress=3)
        meta = dict(self.metadata, feature_columns=self.feature_columns)
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: str | Path) -> "CandidateScorer":
        d = Path(directory)
        if not (d / "model.joblib").exists() or not (d / "metadata.json").exists():
            raise ModelArtifactError(f"No model artifact in {d} (expected model.joblib + metadata.json)")
        meta = json.loads((d / "metadata.json").read_text())
        pipe = joblib.load(d / "model.joblib")
        log.info("Loaded model %s from %s", meta.get("model_name"), d)
        return cls(pipe, meta["feature_columns"], meta)
