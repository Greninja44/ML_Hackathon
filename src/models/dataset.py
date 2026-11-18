"""Build (and cache) labelled candidate-pair tables for a development partition.

Output per partition:

* ``pairs``  – one row per (trip_id, obs, candidate arc) with FEATURE_COLUMNS,
  ``label`` (1 if the candidate is the reference arc) and ``label_residual_m``;
* ``obs``    – one row per labelled observation: trip_id, obs, label_arc,
  residual_m, n_candidates, label_in_candidates. Needed so Top-1 accuracy also
  counts observations whose reference arc was never a candidate.

Only ``train`` and ``student_test`` can be built here (``iter_partition``
refuses ``eval_test``).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import resolve
from src.data.io import TripData, iter_partition
from src.data.prepare import prepare_labelled

log = logging.getLogger(__name__)


@dataclass
class PartitionTables:
    pairs: pd.DataFrame
    obs: pd.DataFrame
    trip_report: pd.DataFrame


def _cache_key(cfg: dict, partition: str) -> str:
    relevant = {k: cfg[k] for k in ("network", "candidates")}
    relevant["data"] = cfg["data"]["raw_dir"]
    relevant["v"] = 1
    h = hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]
    return f"{partition}_{h}"


def _process(trip: TripData, cfg: dict):
    prep, lab = prepare_labelled(trip, cfg)
    counts = prep.cands.counts()
    hit = np.zeros(len(counts), bool)
    pos = prep.features["label"].to_numpy() == 1
    hit[prep.cands.obs[pos]] = True
    obs = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs": np.arange(len(counts)),
        "label_arc": lab.label_arc,
        "residual_m": lab.alignment.residual_m,
        "n_candidates": counts,
        "label_in_candidates": hit,
    })
    report = {"trip_id": trip.trip_id, **prep.track.report,
              "n_pairs": len(prep.features),
              "invalid_route_arcs": lab.alignment.invalid_route_arcs,
              "median_label_residual_m": float(np.nanmedian(lab.alignment.residual_m)) if len(counts) else np.nan,
              "candidate_miss_rate": float(1 - hit[lab.label_arc >= 0].mean()) if (lab.label_arc >= 0).any() else np.nan}
    return prep.features, obs, report


def build_partition(cfg: dict, partition: str, use_cache: bool = True, n_jobs: int = -1) -> PartitionTables:
    cache_dir = resolve(cfg["data"]["cache_dir"])
    key = _cache_key(cfg, partition)
    files = {n: cache_dir / f"{key}_{n}.pkl" for n in ("pairs", "obs", "report")}
    if use_cache and all(f.exists() for f in files.values()):
        log.info("Loading cached feature tables for %s (%s)", partition, key)
        return PartitionTables(pd.read_pickle(files["pairs"]), pd.read_pickle(files["obs"]),
                               pd.read_pickle(files["report"]))

    trips = [t for t in iter_partition(cfg, partition) if t.has_route]
    if not trips:
        raise FileNotFoundError(f"No labelled trips found for partition {partition!r}. "
                                f"Put the raw data under {resolve(cfg['data']['raw_dir'])}/{partition}/")
    log.info("Building features for %d %s trips", len(trips), partition)
    results = Parallel(n_jobs=n_jobs)(delayed(_process)(t, cfg) for t in trips)
    pairs = pd.concat([r[0] for r in results], ignore_index=True)
    obs = pd.concat([r[1] for r in results], ignore_index=True)
    report = pd.DataFrame([r[2] for r in results])
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_pickle(files["pairs"])
    obs.to_pickle(files["obs"])
    report.to_pickle(files["report"])
    return PartitionTables(pairs, obs, report)
