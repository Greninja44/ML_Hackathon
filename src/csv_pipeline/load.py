"""Load and validate Person 1's preprocessed CSVs.

This is the *CSV workflow*, deliberately separate from the raw-data workflow
(`src/data/`). The CSVs contain no coordinates or road geometry, so nothing in
this package can draw maps or compute network distances.

Verified schema (see docs/csv_data_report.md for evidence):

``<partition>_features.csv`` — one row per (GPS point, candidate arc) pair
    partition, trip_id, gps_index (0..n-1 per trip), timestamp [s],
    arc_id (per-trip id), from_node_id, to_node_id, distance_m (point→arc),
    gps_distance_m, gps_bearing, time_gap_sec, gps_speed_mps (per point,
    relative to previous point), road_length_m, road_bearing,
    direction_difference (|gps_bearing − road_bearing| wrapped to [0,180]).
    Candidates of a point are sorted by distance_m; at most 5 per point.

``<partition>_labels.csv`` — the reference route, one row per route step
    partition, trip_id, route_position (0..L-1), true_arc_id.
    This is an ordered arc sequence, NOT a per-point label.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve

log = logging.getLogger(__name__)

FEATURE_COLS = ["partition", "trip_id", "gps_index", "timestamp", "arc_id", "from_node_id", "to_node_id",
                "distance_m", "gps_distance_m", "gps_bearing", "time_gap_sec", "gps_speed_mps",
                "road_length_m", "road_bearing", "direction_difference"]
LABEL_COLS = ["partition", "trip_id", "route_position", "true_arc_id"]
POINT_COLS = ["timestamp", "gps_distance_m", "gps_bearing", "time_gap_sec", "gps_speed_mps"]
ALLOWED = ("train", "student_test")  # evaluator data is never part of this workflow


class CSVSchemaError(ValueError):
    pass


@dataclass
class CSVPartition:
    partition: str
    pairs: pd.DataFrame  # sorted by trip_id, gps_index, distance_m
    route: pd.DataFrame  # sorted by trip_id, route_position
    checks: dict


def csv_dir(cfg: dict) -> Path:
    return resolve(cfg.get("csv_data", {}).get("dir", "data/person1"))


def load_partition(cfg: dict, partition: str, validate: bool = True) -> CSVPartition:
    if partition not in ALLOWED:
        raise PermissionError(f"CSV workflow only reads {ALLOWED}, not {partition!r}")
    d = csv_dir(cfg)
    fp, lp = d / f"{partition}_features.csv", d / f"{partition}_labels.csv"
    for p in (fp, lp):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (expected Person 1's CSVs in {d})")
    pairs = pd.read_csv(fp, dtype={"trip_id": str, "partition": str})
    route = pd.read_csv(lp, dtype={"trip_id": str, "partition": str})
    if list(pairs.columns) != FEATURE_COLS:
        raise CSVSchemaError(f"{fp.name}: unexpected columns {list(pairs.columns)}")
    if list(route.columns) != LABEL_COLS:
        raise CSVSchemaError(f"{lp.name}: unexpected columns {list(route.columns)}")
    pairs = pairs.sort_values(["trip_id", "gps_index", "distance_m"], kind="stable").reset_index(drop=True)
    route = route.sort_values(["trip_id", "route_position"], kind="stable").reset_index(drop=True)
    checks = validate_partition(pairs, route, partition) if validate else {}
    return CSVPartition(partition, pairs, route, checks)


def validate_partition(pairs: pd.DataFrame, route: pd.DataFrame, partition: str) -> dict:
    """Raise on structural violations; return descriptive checks."""
    errors = []
    if not (pairs["partition"] == partition).all() or not (route["partition"] == partition).all():
        errors.append("partition column does not match file")
    if pairs.isna().any().any() or route.isna().any().any():
        errors.append("missing values present")
    if pairs.duplicated(["trip_id", "gps_index", "arc_id"]).any():
        errors.append("duplicate (trip, point, arc) rows")
    if set(pairs["trip_id"]) != set(route["trip_id"]):
        errors.append("feature and label files cover different trips")
    pts = pairs.groupby(["trip_id", "gps_index"])[POINT_COLS].nunique()
    if (pts > 1).any().any():
        errors.append("per-point columns vary across candidates of the same point")
    arc = pairs.groupby(["trip_id", "arc_id"])[["from_node_id", "to_node_id", "road_length_m"]].nunique()
    if (arc > 1).any().any():
        errors.append("an arc id has inconsistent nodes/length within a trip")
    idx = pairs.groupby("trip_id")["gps_index"].agg(["min", "max", "nunique"])
    gaps = int((idx["max"] + 1 - idx["nunique"]).sum())
    rp = route.groupby("trip_id")["route_position"].apply(lambda s: bool((s.values == np.arange(len(s))).all()))
    if not rp.all():
        errors.append("route_position is not 0..L-1 for every trip")
    if errors:
        raise CSVSchemaError(f"{partition}: " + "; ".join(errors))
    n_cand = pairs.groupby(["trip_id", "gps_index"]).size()
    return {
        "partition": partition,
        "n_trips": int(pairs["trip_id"].nunique()),
        "n_pairs": int(len(pairs)),
        "n_points": int(len(n_cand)),
        "candidates_per_point": n_cand.value_counts().sort_index().to_dict(),
        "gps_index_gaps": gaps,
        "route_steps": int(len(route)),
        "route_len_median": float(route.groupby("trip_id").size().median()),
    }
