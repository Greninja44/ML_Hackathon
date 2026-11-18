from __future__ import annotations

import logging
from pathlib import Path

from src.config import resolve

FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(name: str, results_dir: str = "results", level=logging.INFO) -> Path:
    log_dir = resolve(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(logging.Formatter(FMT))
        root.addHandler(h)
    return path
