"""Central configuration.

Values come from ``configs/pipeline.json`` (or the file named by the
``PATHSNAP_CONFIG`` environment variable). A few deployment-specific values can
be overridden with environment variables so nothing machine-specific is
hard-coded:

    PATHSNAP_CONFIG       path to an alternative JSON config
    PATHSNAP_DATA_DIR     root directory holding the partition folders
    PATHSNAP_MODEL_DIR    directory holding the final model artifact
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.json"

PARTITIONS = ("train", "student_test", "eval_test")
# Partitions that training, feature selection, model selection, HPO and
# decoder tuning are permitted to read (Dataset Exploration §7, MT §7/§9).
DEVELOPMENT_PARTITIONS = ("train", "student_test")


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("PATHSNAP_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if os.environ.get("PATHSNAP_DATA_DIR"):
        cfg["data"]["raw_dir"] = os.environ["PATHSNAP_DATA_DIR"]
    if os.environ.get("PATHSNAP_MODEL_DIR"):
        cfg["paths"]["final_model_dir"] = os.environ["PATHSNAP_MODEL_DIR"]
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def partition_dir(cfg: dict, partition: str) -> Path:
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition {partition!r}; expected one of {PARTITIONS}")
    return resolve(cfg["data"]["raw_dir"]) / cfg["data"]["partitions"][partition]
