"""End-to-end checks on the real dataset. Skipped until the data is in data/raw/."""
import numpy as np
import pytest

from src.config import load_config
from src.data.io import iter_partition, list_partition
from src.data.validation import validate_trip
from src.map_matching.reconstruction import RouteReconstructor

CFG = load_config()
HAVE_DATA = bool(list_partition(CFG, "student_test"))
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="real dataset not present in data/raw/student_test")


def test_real_student_trip_end_to_end():
    trip = next(iter_partition(CFG, "student_test"))
    v = validate_trip(trip)
    assert v["n_obs"] > 0 and v["n_arcs"] > 0
    res = RouteReconstructor(CFG).reconstruct(trip.inputs(), "hmm")
    assert len(res.pred_arc) > 0
    assert (res.pred_arc >= 0).mean() > 0.9
    assert np.isfinite(res.timings_ms["total_ms"])


def test_dataset_loader_facade_and_csv_raw_correspondence():
    """Raw .track rows correspond 1:1 to Person 1's CSV gps_index (checked on one student trip)."""
    from src.data import dataset_loader as dl
    ids = dl.raw_trip_ids("student_test")
    assert len(ids) == 10
    trip = dl.load_raw("student_test", ids[0])
    with pytest.raises(PermissionError):
        dl.load_raw("eval_test", ids[0])
    try:
        csv = dl.load_csv("student_test")
    except FileNotFoundError:
        pytest.skip("Person 1 CSVs not present")
    pts = csv.pairs[csv.pairs.trip_id == ids[0]]["gps_index"]
    assert pts.max() + 1 == len(trip.track) == pts.nunique()
    route = csv.route[csv.route.trip_id == ids[0]]["true_arc_id"].to_numpy()
    assert np.array_equal(route, trip.route)
