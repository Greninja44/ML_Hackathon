"""Topology-only HMM for the CSV workflow.

The CSVs give each candidate arc's from/to node ids but no coordinates, so we
can tell whether two arcs *connect* but not how far apart they are along the
network. The full distance-based HMM (src/map_matching/hmm.py) needs the raw
files. This reduced HMM is therefore an approximation, stated as such in the
report:

Hidden state     a candidate arc of the point (candidate→arc id preserved).
Emission         log p from the classifier (``ml``), or a Gaussian on
                 distance_m (``gaussian``, no ML), floored at ``min_log``.
Transition       a → b between consecutive points:
                   same arc                                 0
                   b follows a (to_node(a) == from_node(b))  −c_next
                   b is a's reverse twin (U-turn)            −c_uturn
                   otherwise (connection unknown/indirect)   −c_jump
                 The CSV only lists the ≤5 nearest arcs per point, so an
                 indirect path through arcs never listed as candidates cannot be
                 verified: it is penalised, not forbidden. With a time gap Δt > 1 s
                 the penalties are divided by Δt (more movement is plausible).
Decoding         log-space Viterbi (src/map_matching/viterbi.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.map_matching.viterbi import viterbi


@dataclass
class TopoHMMParams:
    emission: str = "ml"  # "ml" | "gaussian" | "hybrid"
    sigma_m: float = 8.0
    ml_weight: float = 1.0
    c_next: float = 0.5
    c_uturn: float = 6.0
    c_jump: float = 4.0
    min_log: float = -30.0


def emissions(dist: np.ndarray, proba: np.ndarray | None, p: TopoHMMParams) -> np.ndarray:
    gauss = -0.5 * (dist / p.sigma_m) ** 2
    if proba is None or p.emission == "gaussian":
        e = gauss
    else:
        logp = np.log(np.clip(proba, np.exp(p.min_log), 1.0))
        e = logp if p.emission == "ml" else p.ml_weight * logp + (1 - p.ml_weight) * gauss
    return np.maximum(e, p.min_log)


def decode_trip(trip_pairs: pd.DataFrame, proba: np.ndarray | None, p: TopoHMMParams) -> np.ndarray:
    """trip_pairs sorted by gps_index, distance_m. Returns the chosen row position (into trip_pairs) per point."""
    pts = trip_pairs["gps_index"].to_numpy()
    starts = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1]])
    offsets = np.r_[starts, len(pts)]
    arc = trip_pairs["arc_id"].to_numpy()
    fr = trip_pairs["from_node_id"].to_numpy()
    to = trip_pairs["to_node_id"].to_numpy()
    gap = trip_pairs["time_gap_sec"].to_numpy()
    e = emissions(trip_pairs["distance_m"].to_numpy(float), proba, p)
    em = [e[offsets[i]:offsets[i + 1]] for i in range(len(starts))]

    def trans(t: int) -> np.ndarray:
        a = slice(offsets[t - 1], offsets[t])
        b = slice(offsets[t], offsets[t + 1])
        same = arc[a][:, None] == arc[b][None, :]
        nxt = to[a][:, None] == fr[b][None, :]
        uturn = (to[a][:, None] == fr[b][None, :]) & (fr[a][:, None] == to[b][None, :])
        scale = max(1.0, float(gap[offsets[t]]))
        A = np.full(same.shape, -p.c_jump / scale)
        A[nxt] = -p.c_next / scale
        A[uturn] = -p.c_uturn / scale
        A[same] = 0.0
        return A

    res = viterbi(em, trans)
    return np.where(res.state >= 0, offsets[:-1] + res.state, -1)


def argmax_trip(trip_pairs: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Classifier-only decision: highest-scoring candidate of each point (row positions)."""
    pts = trip_pairs["gps_index"].to_numpy()
    starts = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1]])
    ends = np.r_[starts[1:], len(score)]
    return np.array([s + int(np.argmax(score[s:e])) for s, e in zip(starts, ends)], dtype=np.int64)


# ---------------------------------------------------------------- route metrics
def collapse(seq) -> list[int]:
    out: list[int] = []
    for a in seq:
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def _edit(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def route_metrics(trip_pairs: pd.DataFrame, chosen: np.ndarray, route: np.ndarray, label_arc: np.ndarray) -> dict:
    arc = trip_pairs["arc_id"].to_numpy()
    fr = dict(zip(arc, trip_pairs["from_node_id"].to_numpy()))
    to = dict(zip(arc, trip_pairs["to_node_id"].to_numpy()))
    length = dict(zip(arc, trip_pairs["road_length_m"].to_numpy()))
    pred = arc[chosen]
    seq = collapse(pred)
    trans = list(zip(seq[:-1], seq[1:]))
    cont = float(np.mean([to[a] == fr[b] for a, b in trans])) if trans else 1.0
    ref = collapse(route)
    ps, rs = set(seq), set(ref)
    inter = ps & rs
    prec = len(inter) / len(ps) if ps else 0.0
    rec = len(inter) / len(rs) if rs else 0.0
    # length recall is computed over route arcs whose length is known (appear as candidates)
    known = [a for a in rs if a in length]
    lp = sum(length[a] for a in ps)
    return {
        "top1": float(np.mean(pred == label_arc)),
        "continuity": cont,
        "seq_precision": prec, "seq_recall": rec,
        "seq_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "length_precision": sum(length[a] for a in inter) / lp if lp else 0.0,
        "length_recall": (sum(length[a] for a in inter if a in length) / sum(length[a] for a in known)) if known else 0.0,
        "edit_similarity": 1 - _edit(seq, ref) / max(len(seq), len(ref), 1),
        "n_transitions": len(trans),
    }
