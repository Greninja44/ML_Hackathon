"""Observation- and route-level metrics (Models & Training Guidelines §5).

Definitions used (recorded in PROJECT_REQUIREMENTS.md §7):

* **Top-1 accuracy** — fraction of labelled observations whose predicted arc
  equals the reference arc. Observations with no prediction count as wrong.
* **Candidate miss rate** — fraction of labelled observations whose reference
  arc is not among the generated candidates (upper bound on Top-1 = 1 − miss).
* **Route continuity** — over consecutive observations whose predicted arcs
  differ, the fraction where the second arc is directly reachable from the
  first (shares the connecting node; ``max_hops`` > 1 allows short gaps).
* **Sequence-level** — compare the de-duplicated predicted arc sequence with
  the reference route: set precision / recall / F1, length-weighted
  precision / recall, and normalised edit-distance similarity.
"""
from __future__ import annotations

import numpy as np

from src.data.network import RoadNetwork


def top1_accuracy(pred_arc: np.ndarray, label_arc: np.ndarray) -> float:
    mask = label_arc >= 0
    if mask.sum() == 0:
        return float("nan")
    return float((pred_arc[mask] == label_arc[mask]).mean())


def candidate_miss_rate(cand_obs: np.ndarray, cand_arc: np.ndarray, label_arc: np.ndarray) -> float:
    mask = label_arc >= 0
    if mask.sum() == 0:
        return float("nan")
    hit = np.zeros(len(label_arc), bool)
    ok = cand_arc == label_arc[cand_obs]
    hit[cand_obs[ok]] = True
    return float(1.0 - hit[mask].mean())


def _reachable(net: RoadNetwork, a: int, b: int, max_hops: int) -> bool:
    frontier = {int(net.arcs[a, 1])}
    if not net.directed:
        frontier.add(int(net.arcs[a, 0]))
    seen = set(frontier)
    for hop in range(max_hops):
        fb, tb = int(net.arcs[b, 0]), int(net.arcs[b, 1])
        if fb in frontier or (not net.directed and tb in frontier):
            return True
        if hop == max_hops - 1:
            break
        nxt = set()
        for n in frontier:
            for arc in net.out_arcs[n]:
                nxt.add(int(net.arcs[arc, 1]))
            if not net.directed:
                for arc in net.in_arcs[n]:
                    nxt.add(int(net.arcs[arc, 0]))
        frontier = nxt - seen
        seen |= nxt
    return False


def route_continuity(pred_arc: np.ndarray, net: RoadNetwork, max_hops: int = 1) -> dict:
    p = pred_arc[pred_arc >= 0]
    trans = [(int(a), int(b)) for a, b in zip(p[:-1], p[1:]) if a != b]
    if not trans:
        return {"route_continuity": 1.0 if len(p) else float("nan"), "n_transitions": 0, "n_discontinuities": 0}
    ok = sum(_reachable(net, a, b, max_hops) for a, b in trans)
    return {"route_continuity": ok / len(trans), "n_transitions": len(trans),
            "n_discontinuities": len(trans) - ok}


def collapse(seq) -> list[int]:
    out: list[int] = []
    for a in seq:
        a = int(a)
        if a >= 0 and (not out or out[-1] != a):
            out.append(a)
    return out


def _edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def sequence_metrics(pred_route: list[int], ref_route: np.ndarray, net: RoadNetwork) -> dict:
    ref = collapse(ref_route)
    pred = collapse(pred_route)
    ps, rs = set(pred), set(ref)
    inter = ps & rs
    prec = len(inter) / len(ps) if ps else 0.0
    rec = len(inter) / len(rs) if rs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    L = net.arc_len
    len_inter = float(sum(L[a] for a in inter))
    len_p = float(sum(L[a] for a in ps))
    len_r = float(sum(L[a] for a in rs if 0 <= a < len(L)))
    ed = _edit_distance(pred, ref)
    return {
        "seq_precision": prec,
        "seq_recall": rec,
        "seq_f1": f1,
        "length_precision": len_inter / len_p if len_p else 0.0,
        "length_recall": len_inter / len_r if len_r else 0.0,
        "edit_similarity": 1.0 - ed / max(len(pred), len(ref), 1),
    }
