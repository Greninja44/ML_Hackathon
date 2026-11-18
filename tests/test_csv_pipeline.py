"""Tests for the CSV workflow using small hand-built tables (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from src.csv_pipeline.features import CSV_FEATURES, build_features
from src.csv_pipeline.hmm import TopoHMMParams, argmax_trip, decode_trip, route_metrics
from src.csv_pipeline.labels import align_trip, pair_labels
from src.csv_pipeline.load import CSVSchemaError, validate_partition

# Network: 1 -(a10)-> 2 -(a11)-> 3 -(a12)-> 4, plus reverse twins a20 (2->1), a21 (3->2), and a side road a30 (2->5).
ARCS = {10: (1, 2), 11: (2, 3), 12: (3, 4), 20: (2, 1), 21: (3, 2), 30: (2, 5)}


def _pairs(rows):
    """rows: list of (gps_index, arc_id, distance_m)."""
    out = []
    for i, a, d in rows:
        f, t = ARCS[a]
        out.append({"partition": "train", "trip_id": "T", "gps_index": i, "timestamp": float(i), "arc_id": a,
                    "from_node_id": f, "to_node_id": t, "distance_m": d, "gps_distance_m": 10.0,
                    "gps_bearing": 90.0, "time_gap_sec": 1.0 if i else 0.0, "gps_speed_mps": 10.0,
                    "road_length_m": 100.0, "road_bearing": 90.0 if a in (10, 11, 12) else 270.0,
                    "direction_difference": 0.0 if a in (10, 11, 12) else 180.0})
    return pd.DataFrame(out).sort_values(["trip_id", "gps_index", "distance_m"], kind="stable").reset_index(drop=True)


ROUTE = np.array([10, 11, 12])
ROWS = [
    (0, 10, 2.0), (0, 20, 2.0), (0, 30, 40.0),
    (1, 10, 3.0), (1, 20, 3.0), (1, 11, 9.0),
    (2, 11, 4.0), (2, 21, 4.0), (2, 10, 8.0),   # nearest on-route arc is 11
    (3, 12, 3.0), (3, 11, 5.0),
    (4, 30, 6.0), (4, 20, 9.0),                  # route arc not a candidate -> miss
    (5, 12, 2.0),
]


def test_alignment_is_monotone_and_marks_misses():
    obs = align_trip(_pairs(ROWS), ROUTE)
    assert obs["route_pos"].is_monotonic_increasing
    assert obs["label_arc"].tolist()[:4] == [10, 10, 11, 12]
    assert not obs.loc[obs.gps_index == 4, "label_in_candidates"].item()   # kept, counted as a miss
    assert obs["label_in_candidates"].sum() == 5


def test_alignment_respects_route_order_over_nearest():
    # point 1 is slightly nearer to arc 12 than 11, but route order (after a confident 11) forbids going back
    rows = [(0, 11, 1.0), (1, 12, 4.0), (1, 11, 4.5), (2, 11, 1.0), (3, 12, 1.0)]
    obs = align_trip(_pairs(rows), np.array([11, 12]))
    assert obs["label_arc"].tolist() == [11, 11, 11, 12]


def test_pair_labels_one_positive_per_point():
    p = _pairs(ROWS)
    obs = align_trip(p, ROUTE).assign(trip_id="T")
    y = pair_labels(p, obs)
    per_point = pd.Series(y).groupby(p["gps_index"]).sum()
    assert per_point.max() == 1 and per_point.loc[4] == 0


def test_features_finite_and_independent_of_route():
    p = _pairs(ROWS)
    f = build_features(p)
    assert np.isfinite(f[CSV_FEATURES].to_numpy(float)).all()
    assert f.loc[(f.gps_index == 0) & (f.arc_id == 10), "has_reverse_twin"].item() == 1
    assert f.loc[(f.gps_index == 1) & (f.arc_id == 10), "prev_link"].item() == 2   # same as prev nearest
    assert f.loc[(f.gps_index == 2) & (f.arc_id == 11), "prev_link"].item() == 1   # follows prev nearest (10)
    assert "label" not in f.columns


def test_topology_hmm_fixes_isolated_wrong_pick():
    p = _pairs([(0, 10, 2.0), (0, 20, 2.0), (1, 10, 2.0), (1, 20, 1.5), (2, 10, 2.0), (2, 20, 2.0)])
    # classifier slightly prefers the reverse twin at point 1; transitions should keep arc 10
    want = {(0, 10): 0.8, (0, 20): 0.2, (1, 10): 0.45, (1, 20): 0.55, (2, 10): 0.8, (2, 20): 0.2}
    proba = np.array([want[(i, a)] for i, a in zip(p["gps_index"], p["arc_id"])])
    cls = argmax_trip(p, proba)
    assert p["arc_id"].to_numpy()[cls].tolist() == [10, 20, 10]
    hmm = decode_trip(p, proba, TopoHMMParams())
    assert p["arc_id"].to_numpy()[hmm].tolist() == [10, 10, 10]
    m = route_metrics(p, hmm, np.array([10]), np.array([10, 10, 10]))
    assert m["top1"] == 1.0 and m["continuity"] == 1.0


def test_validation_rejects_inconsistent_points():
    p = _pairs(ROWS)
    r = pd.DataFrame({"partition": "train", "trip_id": "T", "route_position": [0, 1, 2], "true_arc_id": ROUTE})
    assert validate_partition(p, r, "train")["n_points"] == 6
    bad = p.copy()
    bad.loc[0, "gps_speed_mps"] = 99.0
    with pytest.raises(CSVSchemaError):
        validate_partition(bad, r, "train")


def test_csv_workflow_refuses_eval_partition():
    from src.config import load_config
    from src.csv_pipeline.load import load_partition
    with pytest.raises(PermissionError):
        load_partition(load_config(), "eval_test")
