"""Derive per-observation reference arcs from the trip-level ``.route``.

The ``.route`` file lists the arcs traversed, in order, but does not say which
GPS observation belongs to which arc (Dataset Exploration §3: "Do not assume
that a row in a GPS file corresponds directly to a row in a road-network
file").

We assign each observation ``i`` a route position ``k(i)`` such that ``k`` is
non-decreasing in time (the vehicle cannot go back along its route) and the
sum of point-to-arc distances is minimal. This is solved exactly by dynamic
programming in O(N·L):

    D[i, k] = dist(i, route[k]) + min_{k' <= k} D[i-1, k']

The label of observation ``i`` is ``route[k(i)]``. Alignment residuals
(distance from point to its label arc) are returned so label quality can be
reported — the reference route is not treated as error-free (DE §6).

Labels are only ever used for training/evaluation, never as model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.network import RoadNetwork
from src.geo import project_points_to_segments


@dataclass
class RouteAlignment:
    label_arc: np.ndarray  # (n_obs,) arc id per observation (-1 if impossible)
    route_pos: np.ndarray  # (n_obs,) index into route
    residual_m: np.ndarray  # (n_obs,) distance from observation to its label arc
    invalid_route_arcs: int  # route entries that are not valid arc ids


def align_route(xy: np.ndarray, route: np.ndarray, net: RoadNetwork) -> RouteAlignment:
    n = len(xy)
    route = np.asarray(route, dtype=np.int64)
    valid = (route >= 0) & (route < net.n_arcs)
    valid[valid] &= net.valid[route[valid]]
    n_invalid = int((~valid).sum())
    r = route[valid]
    if n == 0 or len(r) == 0:
        return RouteAlignment(np.full(n, -1), np.full(n, -1), np.full(n, np.nan), n_invalid)

    L = len(r)
    a, b = net.arc_a[r], net.arc_b[r]
    # distance matrix (n, L), computed row-block-wise to bound memory
    dist = np.empty((n, L))
    block = max(1, 2_000_000 // L)
    for s in range(0, n, block):
        e = min(n, s + block)
        p = np.repeat(xy[s:e], L, axis=0)
        d, _ = project_points_to_segments(p, np.tile(a, (e - s, 1)), np.tile(b, (e - s, 1)))
        dist[s:e] = d.reshape(e - s, L)

    D = np.empty((n, L))
    back = np.empty((n, L), dtype=np.int64)
    D[0] = dist[0]
    back[0] = np.arange(L)
    for i in range(1, n):
        prev = D[i - 1]
        # running argmin of prev over k' <= k
        cummin = np.minimum.accumulate(prev)
        is_new = np.concatenate([[True], prev[1:] < cummin[:-1]])
        arg = np.maximum.accumulate(np.where(is_new, np.arange(L), 0))
        D[i] = dist[i] + cummin
        back[i] = arg

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(D[-1]))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return RouteAlignment(r[pos], pos, dist[np.arange(n), pos], n_invalid)
