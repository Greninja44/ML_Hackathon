"""Derive a per-point reference arc from the ordered route (CSV workflow).

Why not "candidate appears anywhere in the route"?  A point near a junction
usually has two or more candidates on the route (the arc it is on and the next
one), and loops put the same arc at several route positions. That label would
mark several candidates of one point as correct and is not a map-matching
target.

What we do instead — monotone alignment
---------------------------------------
Each GPS point i is assigned a route position k(i), non-decreasing over time
(the vehicle drives the route in order), minimising Σ cost(i, k(i)):

* cost(i, k) = distance_m(i, route[k])        if route[k] is a candidate of i
* cost(i, k) = lower bound of that distance    otherwise

The CSV only stores distances to a point's ≤5 nearest arcs. If route[k] is not
among them, its true distance is *at least* the distance of the point's
farthest candidate (when the point has the full 5 candidates) — so we use that
lower bound plus ``outside_margin_m``. When a point has fewer than 5 candidates
the search radius is unknown, so ``unknown_radius_m`` is used.

The label is route[k(i)]. If it is not among the point's candidates, the point
is a **candidate miss**: it stays in the evaluation and counts as wrong.

The alignment is an *inference*, so its robustness is measured
(``stability_report``) by re-running it with different cost settings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_CANDIDATES = 5


@dataclass
class AlignParams:
    outside_margin_m: float = 5.0
    unknown_radius_m: float = 50.0


def align_trip(pairs: pd.DataFrame, route: np.ndarray, params: AlignParams = AlignParams()) -> pd.DataFrame:
    """pairs: one trip's rows sorted by gps_index, distance_m. Returns one row per point."""
    pts = np.sort(pairs["gps_index"].unique())
    n, L = len(pts), len(route)
    pos_of_point = {p: i for i, p in enumerate(pts)}
    pi = pairs["gps_index"].map(pos_of_point).to_numpy()
    n_cand = np.bincount(pi, minlength=n)
    far = pairs.groupby("gps_index")["distance_m"].max().reindex(pts).to_numpy()
    outside = np.where(n_cand >= MAX_CANDIDATES, far + params.outside_margin_m,
                       np.maximum(far, params.unknown_radius_m) + params.outside_margin_m)

    # route positions of each arc (arcs can repeat in loops)
    positions: dict[int, list[int]] = {}
    for k, a in enumerate(route):
        positions.setdefault(int(a), []).append(k)

    # sparse candidate costs: for each point, (route position, distance)
    sparse_k, sparse_d, sparse_i = [], [], []
    for i_pt, arc, d in zip(pi, pairs["arc_id"].to_numpy(), pairs["distance_m"].to_numpy()):
        for k in positions.get(int(arc), ()):
            sparse_i.append(i_pt); sparse_k.append(k); sparse_d.append(d)
    sparse_i, sparse_k, sparse_d = map(np.asarray, (sparse_i, sparse_k, sparse_d))
    order = np.argsort(sparse_i, kind="stable")
    sparse_i, sparse_k, sparse_d = sparse_i[order], sparse_k[order], sparse_d[order]
    starts = np.searchsorted(sparse_i, np.arange(n + 1))

    back_dtype = np.uint16 if L < 2 ** 16 else np.uint32
    back = np.empty((n, L), dtype=back_dtype)
    prev = None
    ar = np.arange(L)
    for i in range(n):
        row = np.full(L, outside[i])
        s, e = starts[i], starts[i + 1]
        if e > s:
            row[sparse_k[s:e]] = sparse_d[s:e]
        if prev is None:
            cur = row
            back[i] = ar
        else:
            cummin = np.minimum.accumulate(prev)
            is_new = np.empty(L, bool)
            is_new[0] = True
            is_new[1:] = prev[1:] < cummin[:-1]
            back[i] = np.maximum.accumulate(np.where(is_new, ar, 0))
            cur = row + cummin
        prev = cur
    k_of = np.empty(n, dtype=np.int64)
    k_of[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        k_of[i - 1] = back[i, k_of[i]]

    label = route[k_of]
    cand = pairs.assign(_pi=pi)
    is_lab = cand["arc_id"].to_numpy() == label[pi]
    hit = np.zeros(n, bool)
    hit[pi[is_lab]] = True
    resid = np.full(n, np.nan)
    resid[pi[is_lab]] = pairs["distance_m"].to_numpy()[is_lab]
    nearest_on_route = np.full(n, -1)
    on_route = np.isin(pairs["arc_id"].to_numpy(), route)
    first = pd.Series(np.where(on_route, np.arange(len(pairs)), np.iinfo(np.int64).max)).groupby(pi).min()
    ok = first.to_numpy() < len(pairs)
    nearest_on_route[first.index.to_numpy()[ok]] = pairs["arc_id"].to_numpy()[first.to_numpy()[ok]]
    return pd.DataFrame({
        "gps_index": pts, "route_pos": k_of, "label_arc": label, "label_in_candidates": hit,
        "label_distance_m": resid, "n_candidates": n_cand, "nearest_on_route_arc": nearest_on_route,
    })


def align_partition(pairs: pd.DataFrame, route: pd.DataFrame, params: AlignParams = AlignParams()) -> pd.DataFrame:
    out = []
    for tid, g in pairs.groupby("trip_id", sort=True):
        r = route.loc[route["trip_id"] == tid, "true_arc_id"].to_numpy()
        a = align_trip(g, r, params)
        a.insert(0, "trip_id", tid)
        out.append(a)
    return pd.concat(out, ignore_index=True)


def pair_labels(pairs: pd.DataFrame, obs: pd.DataFrame) -> np.ndarray:
    m = pairs[["trip_id", "gps_index"]].merge(obs[["trip_id", "gps_index", "label_arc"]],
                                             on=["trip_id", "gps_index"], how="left")
    return (pairs["arc_id"].to_numpy() == m["label_arc"].to_numpy()).astype(int)


def stability_report(pairs: pd.DataFrame, route: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Re-run the alignment with different cost assumptions and measure label agreement."""
    variants = {
        "margin_0m": AlignParams(outside_margin_m=0.0),
        "margin_20m": AlignParams(outside_margin_m=20.0),
        "radius_30m": AlignParams(unknown_radius_m=30.0),
        "radius_100m": AlignParams(unknown_radius_m=100.0),
    }
    rows = []
    for name, p in variants.items():
        alt = align_partition(pairs, route, p)
        agree = (alt["label_arc"].to_numpy() == base["label_arc"].to_numpy())
        both_hit = base["label_in_candidates"].to_numpy()
        rows.append({"variant": name, "label_agreement": float(agree.mean()),
                     "agreement_on_covered_points": float(agree[both_hit].mean()),
                     "miss_rate": float(1 - alt["label_in_candidates"].mean())})
    return pd.DataFrame(rows)
