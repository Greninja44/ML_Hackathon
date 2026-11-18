import numpy as np
import pandas as pd
import pytest

from src.data.io import TripInputs, iter_partition
from src.data.prepare import build_network, prepare_trip
from src.map_matching.candidates import generate_candidates
from src.map_matching.hmm import TransitionModel, emission_scores
from src.map_matching.metrics import collapse, route_continuity, sequence_metrics
from src.map_matching.reconstruction import RouteReconstructor
from src.map_matching.viterbi import viterbi


def _trip(cfg):
    return next(iter_partition(cfg, "train"))


# ------------------------------------------------------------------ Viterbi
def test_viterbi_prefers_consistent_path():
    em = [np.array([0.0, -1.0]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])]
    stay = np.array([[0.0, -10.0], [-10.0, 0.0]])
    res = viterbi(em, lambda t: stay)
    assert res.state.tolist() == [0, 0, 0]  # transitions outweigh one bad emission
    assert res.n_breaks == 0


def test_viterbi_breaks_on_empty_and_impossible():
    em = [np.array([0.0]), np.array([]), np.array([0.0, -1.0]), np.array([0.0])]
    impossible = np.full((2, 1), -np.inf)
    res = viterbi(em, lambda t: impossible if t == 3 else np.zeros((1, 2)))
    assert res.state.tolist() == [0, -1, 0, 0]
    assert res.segment[1] == -1
    assert res.n_breaks == 2
    assert len(set(res.segment[[0, 2, 3]])) == 3


def test_viterbi_empty_input():
    res = viterbi([], lambda t: None)
    assert len(res.state) == 0


def test_viterbi_long_sequence_no_underflow():
    T = 20_000
    em = [np.array([-30.0, -31.0])] * T
    res = viterbi(em, lambda t: np.array([[-20.0, -25.0], [-25.0, -20.0]]))
    assert (res.state == 0).all() and np.isfinite(res.log_score)


# ------------------------------------------------------------------ HMM pieces
def test_emission_modes(cfg):
    trip = _trip(cfg)
    prep = prepare_trip(trip.inputs(), cfg)
    c = prep.cands
    g = emission_scores(c, cfg["hmm"], None)
    assert (g <= 0).all()
    # nearer candidates get higher Gaussian emission
    s = c.for_obs(0)
    assert np.all(np.diff(g[s]) <= 1e-12)
    p = np.full(len(c.arc), 0.5)
    ml = emission_scores(c, cfg["hmm"], p, mode="ml")
    assert np.allclose(ml, np.log(0.5))
    zero = emission_scores(c, cfg["hmm"], np.zeros(len(c.arc)), mode="ml")
    assert np.isfinite(zero).all()  # floored, never -inf
    with pytest.raises(ValueError):
        emission_scores(c, cfg["hmm"], p, mode="nope")


def test_transition_distances(cfg):
    trip = _trip(cfg)
    prep = prepare_trip(trip.inputs(), cfg)
    tm = TransitionModel(prep.net, prep.track, prep.cands, cfg["hmm"])
    a = int(trip.route[0])
    b = int(trip.route[1])
    L = prep.net.arc_len[a]
    assert tm.route_distance(a, 10.0, a, 30.0, 1000) == pytest.approx(20.0)
    # going backwards along a directed arc is not a short move
    assert tm.route_distance(a, 50.0, a, 10.0, 100) == np.inf
    # consecutive route arcs: remaining length of a + offset on b
    assert tm.route_distance(a, 10.0, b, 5.0, 1000) == pytest.approx(L - 10.0 + 5.0)
    A = tm.matrix(1)
    n = prep.cands.counts()
    assert A.shape == (n[0], n[1])
    assert (A <= 0).all()


def test_candidates_limits_and_fallback(cfg):
    trip = _trip(cfg)
    prep = prepare_trip(trip.inputs(), cfg)
    assert prep.cands.counts().max() <= cfg["candidates"]["max_candidates"]
    # a point far from every road only gets candidates through the fallback radius
    tr = prep.track
    far = type(tr)(**{**tr.__dict__, "xy": tr.xy[:1] + np.array([[60.0, 60.0]])})
    cs = generate_candidates(far, prep.net, radius_m=5, max_candidates=3, fallback_radius_m=500)
    assert cs.counts()[0] > 0 and cs.fallback.all()
    cs = generate_candidates(far, prep.net, radius_m=5, max_candidates=3, fallback_radius_m=None)
    assert cs.counts()[0] == 0


# ------------------------------------------------------------------ metrics
def test_route_metrics(cfg):
    trip = _trip(cfg)
    net = build_network(trip.inputs(), cfg)
    r = trip.route
    assert route_continuity(np.repeat(r, 3), net)["route_continuity"] == 1.0
    sm = sequence_metrics(list(r), r, net)
    assert sm["seq_f1"] == 1.0 and sm["edit_similarity"] == 1.0
    assert collapse([3, 3, -1, 4, 4, 3]) == [3, 4, 3]


# ------------------------------------------------------------------ reconstruction edge cases
def _inputs(trip, track):
    return TripInputs(trip.trip_id, track.reset_index(drop=True), trip.nodes, trip.arcs)


@pytest.mark.parametrize("mode", ["hmm", "classifier"])
def test_reconstruction_edge_cases(cfg, mode):
    trip = _trip(cfg)
    rec = RouteReconstructor(cfg)  # no model -> distance fallback
    empty = rec.reconstruct(_inputs(trip, trip.track.iloc[:0]), mode)
    assert len(empty.pred_arc) == 0 and empty.route_arcs == []
    one = rec.reconstruct(_inputs(trip, trip.track.iloc[:1]), mode)
    assert len(one.pred_arc) == 1 and one.pred_arc[0] >= 0
    dup = pd.concat([trip.track.iloc[:20]] * 2).sort_values("t", kind="stable")
    assert len(rec.reconstruct(_inputs(trip, dup), mode).pred_arc) == 20
    t = trip.track.copy()
    t.loc[30, ["lon", "lat"]] = t.loc[30, ["lon", "lat"]] + 0.05  # ~5 km outlier, no candidates
    res = rec.reconstruct(_inputs(trip, t), mode)
    assert res.pred_arc[30] == -1 and (res.pred_arc >= 0).sum() == len(t) - 1
    assert res.emission_source in ("gaussian", "nearest")


def test_hmm_handles_gps_gap(cfg):
    trip = _trip(cfg)
    t = trip.track.copy()
    t = t.drop(index=range(40, 70)).reset_index(drop=True)  # 30 s dropout
    res = RouteReconstructor(cfg).reconstruct(_inputs(trip, t), "hmm")
    assert (res.pred_arc >= 0).all()


def test_hmm_beats_nearest_arc_on_two_way_roads(cfg):
    """On the synthetic grid every road is two directed arcs at identical distance;
    only sequence information can pick the travel direction."""
    from src.data.labels import align_route
    from src.map_matching.metrics import top1_accuracy
    trip = _trip(cfg)
    rec = RouteReconstructor(cfg)
    prep = prepare_trip(trip.inputs(), cfg)
    lab = align_route(prep.track.xy, trip.route, prep.net).label_arc
    near = top1_accuracy(rec.reconstruct(trip.inputs(), "classifier", prepared=prep).pred_arc, lab)
    hmm = top1_accuracy(rec.reconstruct(trip.inputs(), "hmm", prepared=prep).pred_arc, lab)
    assert hmm > near + 0.2
