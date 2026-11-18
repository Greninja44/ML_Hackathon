import numpy as np
import pandas as pd
import pytest

from src.data.features import FEATURE_COLUMNS
from src.data.io import DataFormatError, discover_trips, iter_partition, load_track
from src.data.labels import align_route
from src.data.prepare import attach_labels, build_network, prepare_labelled, prepare_trip
from src.data.preprocess import clean_track
from src.data.validation import validate_trip


def _first(cfg, part="train"):
    return next(iter_partition(cfg, part))


def test_discovery_finds_all_files(syn_root):
    trips = discover_trips(syn_root / "train")
    assert len(trips) == 6
    for files in trips.values():
        assert set(files) == {".track", ".nodes", ".arcs", ".route"}


def test_eval_partition_is_guarded(cfg):
    with pytest.raises(PermissionError):
        next(iter_partition(cfg, "eval_test"))
    assert len(list(iter_partition(cfg, "eval_test", allow_eval=True))) == 3


def test_track_format_errors(tmp_path):
    p = tmp_path / "bad.track"
    p.write_text("1.0 2.0\n3.0 4.0\n")
    with pytest.raises(DataFormatError):
        load_track(p)
    p.write_text("")
    assert load_track(p).empty


def test_validation_reports_problems(cfg):
    trip = _first(cfg)
    ok = validate_trip(trip)
    assert ok["n_issues"] == 0 and ok["route_share_connected"] == 1.0
    trip.arcs = np.vstack([trip.arcs, [[0, 10_000]]])
    t = trip.track.copy()
    t.loc[5, "t"] = t.loc[4, "t"] - 3
    t.loc[7, "lon"] = np.nan
    trip.track = t
    bad = validate_trip(trip)
    assert bad["n_arcs_bad_node_ref"] == 1
    assert bad["n_backward_timestamps"] == 1
    assert bad["n_missing_values"] == 1
    assert bad["n_issues"] >= 3


def test_clean_track_handles_nan_duplicates_and_order(cfg):
    trip = _first(cfg)
    net = build_network(trip.inputs(), cfg)
    t = trip.track.copy()
    t = pd.concat([t, t.iloc[[3]]], ignore_index=True)  # exact duplicate
    t.loc[10, "lat"] = np.nan
    t = t.iloc[::-1].reset_index(drop=True)  # reversed time order
    ct = clean_track(t, net.proj)
    assert ct.report["n_invalid_dropped"] == 1
    assert ct.report["n_duplicates_dropped"] == 1
    assert not ct.report["was_time_sorted"]
    assert np.all(np.diff(ct.t) >= 0)
    assert len(ct) == len(trip.track) - 1


def test_route_alignment_is_monotone_and_close(cfg):
    trip = _first(cfg)
    prep = prepare_trip(trip.inputs(), cfg)
    al = align_route(prep.track.xy, trip.route, prep.net)
    assert np.all(np.diff(al.route_pos) >= 0)
    assert np.median(al.residual_m) < 10  # synthetic noise sigma = 5 m
    assert set(al.label_arc) <= set(trip.route.tolist())


def test_features_do_not_depend_on_route(cfg):
    trip = _first(cfg)
    a = prepare_trip(trip.inputs(), cfg).features
    trip.route = trip.route[::-1].copy()  # a different route must not change features
    b = prepare_trip(trip.inputs(), cfg).features
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["trip_id", "obs", "arc"] + FEATURE_COLUMNS
    assert np.isfinite(a[FEATURE_COLUMNS].to_numpy(float)).all()


def test_labels_one_positive_per_covered_observation(cfg):
    trip = _first(cfg)
    prep, lab = prepare_labelled(trip, cfg)
    per_obs = prep.features.groupby("obs")["label"].sum()
    assert per_obs.max() == 1
    assert per_obs.mean() > 0.95  # candidates contain the reference arc for almost all points


def test_attach_labels_requires_route(cfg):
    trip = _first(cfg)
    prep = prepare_trip(trip.inputs(), cfg)
    trip.route = None
    with pytest.raises(ValueError):
        attach_labels(prep, trip)
