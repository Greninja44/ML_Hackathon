import json

import pytest

from src.config import load_config
from tests.synthetic import write_dataset


@pytest.fixture(scope="session")
def syn_root(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("syn"), n_train=6, n_student=2, n_eval=3, seed=11)


@pytest.fixture(scope="session")
def cfg(syn_root, tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    return load_config(overrides={
        "data": {"raw_dir": str(syn_root), "cache_dir": str(work / "processed")},
        "paths": {"models_dir": str(work / "models"), "results_dir": str(work / "results"),
                  "final_model_dir": str(work / "models" / "final")},
        "training": {"cv_folds": 3, "n_jobs": 1},
    })


@pytest.fixture(scope="session")
def trained_model_dir(cfg):
    """Fit a small model on the synthetic training partition and save it as the final artifact."""
    from src.data.features import FEATURE_COLUMNS
    from src.models.dataset import build_partition
    from src.models.inference import CandidateScorer
    from src.models.registry import default_specs
    from src.models.train import fit_pipeline

    spec = default_specs(1)["random_forest"]
    params = dict(spec.params, n_estimators=30, n_jobs=1)
    train = build_partition(cfg, "train", use_cache=False, n_jobs=1)
    pipe, _ = fit_pipeline(spec, params, train.pairs, cfg["seed"])
    from src.config import resolve
    d = resolve(cfg["paths"]["final_model_dir"])
    CandidateScorer(pipe, FEATURE_COLUMNS, {"model_name": "rf_test"}).save(d)
    (d / "hmm_config.json").write_text(json.dumps(cfg["hmm"]))
    return d
