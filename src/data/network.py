"""Road-network representation for a single trip.

Arcs are straight segments between two nodes (the raw data has no shape
points). Coordinates are projected to metres with :class:`LocalProjection`.

Provides:
* spatial lookup of arcs near a point (KD-tree over densified arc samples),
* bounded shortest-path distances between nodes (Dijkstra, cached),
* path reconstruction as a list of arc ids, used to output a connected route.
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from src.geo import LocalProjection, bearing_deg, project_points_to_segments

log = logging.getLogger(__name__)


@dataclass
class _SPCache:
    cutoff: float
    dist: dict[int, float]
    pred_arc: dict[int, int]


@dataclass
class RoadNetwork:
    proj: LocalProjection
    node_lonlat: np.ndarray
    node_xy: np.ndarray
    arcs: np.ndarray  # (m, 2) from, to
    directed: bool = True
    densify_step_m: float = 10.0
    valid: np.ndarray = field(init=False)
    arc_a: np.ndarray = field(init=False)
    arc_b: np.ndarray = field(init=False)
    arc_len: np.ndarray = field(init=False)
    arc_bearing: np.ndarray = field(init=False)

    def __post_init__(self):
        n = len(self.node_xy)
        arcs = np.asarray(self.arcs, dtype=np.int64).reshape(-1, 2)
        self.arcs = arcs
        self.valid = (arcs >= 0).all(axis=1) & (arcs < n).all(axis=1)
        if (~self.valid).any():
            log.warning("%d arcs reference non-existent nodes and are ignored", int((~self.valid).sum()))
        safe = np.where(self.valid[:, None], arcs, 0)
        self.arc_a = self.node_xy[safe[:, 0]] if n else np.zeros((len(arcs), 2))
        self.arc_b = self.node_xy[safe[:, 1]] if n else np.zeros((len(arcs), 2))
        d = self.arc_b - self.arc_a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = bearing_deg(d[:, 0], d[:, 1])
        self._build_index()
        self._build_adjacency()
        self._sp_cache: dict[int, _SPCache] = {}
        self._reverse_lookup = {(int(f), int(t)): i for i, (f, t) in enumerate(arcs) if self.valid[i]}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_arrays(cls, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                    directed: bool = True, densify_step_m: float = 10.0) -> "RoadNetwork":
        nodes_lonlat = np.asarray(nodes_lonlat, float).reshape(-1, 2)
        node_xy = proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1]) if len(nodes_lonlat) else np.zeros((0, 2))
        return cls(proj, nodes_lonlat, node_xy, arcs, directed, densify_step_m)

    def _build_index(self):
        step = max(self.densify_step_m, 1.0)
        pts, owner = [], []
        for i in np.flatnonzero(self.valid):
            k = max(int(np.ceil(self.arc_len[i] / step)), 1)
            t = np.linspace(0.0, 1.0, k + 1)
            pts.append(self.arc_a[i] + t[:, None] * (self.arc_b[i] - self.arc_a[i]))
            owner.append(np.full(k + 1, i))
        if pts:
            self._sample_xy = np.vstack(pts)
            self._sample_arc = np.concatenate(owner)
            self._tree = cKDTree(self._sample_xy)
        else:
            self._sample_xy = np.zeros((0, 2))
            self._sample_arc = np.zeros(0, dtype=np.int64)
            self._tree = None

    def _build_adjacency(self):
        n = len(self.node_xy)
        self.out_arcs: list[list[int]] = [[] for _ in range(n)]
        self.in_arcs: list[list[int]] = [[] for _ in range(n)]
        for i in np.flatnonzero(self.valid):
            f, t = self.arcs[i]
            self.out_arcs[f].append(int(i))
            self.in_arcs[t].append(int(i))

    # ---------------------------------------------------------------- queries
    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    def reverse_arc(self, arc_id: int) -> int | None:
        f, t = self.arcs[arc_id]
        return self._reverse_lookup.get((int(t), int(f)))

    def arcs_near(self, xy: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arcs within ``radius_m`` of point ``xy``: (arc_ids, distances, t-along-arc), sorted by distance."""
        if self._tree is None:
            return np.zeros(0, int), np.zeros(0), np.zeros(0)
        idx = self._tree.query_ball_point(xy, radius_m + self.densify_step_m / 2 + 1e-6)
        if not idx:
            return np.zeros(0, int), np.zeros(0), np.zeros(0)
        arc_ids = np.unique(self._sample_arc[idx])
        p = np.repeat(np.asarray(xy, float)[None, :], len(arc_ids), axis=0)
        dist, t = project_points_to_segments(p, self.arc_a[arc_ids], self.arc_b[arc_ids])
        keep = dist <= radius_m
        order = np.argsort(dist[keep], kind="stable")
        return arc_ids[keep][order], dist[keep][order], t[keep][order]

    def distance_to_arcs(self, xy: np.ndarray, arc_ids: np.ndarray) -> np.ndarray:
        p = np.repeat(np.asarray(xy, float)[None, :], len(arc_ids), axis=0)
        dist, _ = project_points_to_segments(p, self.arc_a[arc_ids], self.arc_b[arc_ids])
        return dist

    # ---------------------------------------------------------- shortest path
    def _neighbors(self, node: int):
        for a in self.out_arcs[node]:
            yield int(self.arcs[a, 1]), a
        if not self.directed:
            for a in self.in_arcs[node]:
                yield int(self.arcs[a, 0]), a

    def _dijkstra(self, source: int, cutoff: float) -> _SPCache:
        cached = self._sp_cache.get(source)
        if cached is not None and cached.cutoff >= cutoff:
            return cached
        dist = {source: 0.0}
        pred: dict[int, int] = {}
        heap = [(0.0, source)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, a in self._neighbors(u):
                nd = d + self.arc_len[a]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        res = _SPCache(cutoff, dist, pred)
        self._sp_cache[source] = res
        return res

    def node_distance(self, u: int, v: int, cutoff: float) -> float:
        """Shortest network distance u→v in metres, or ``inf`` if > cutoff."""
        if u == v:
            return 0.0
        return self._dijkstra(u, cutoff).dist.get(v, np.inf)

    def node_path(self, u: int, v: int, cutoff: float) -> list[int] | None:
        """Arc ids of the shortest path u→v, or None if unreachable within cutoff."""
        if u == v:
            return []
        sp = self._dijkstra(u, cutoff)
        if v not in sp.dist:
            return None
        path, cur = [], v
        while cur != u:
            a = sp.pred_arc[cur]
            path.append(a)
            f, t = int(self.arcs[a, 0]), int(self.arcs[a, 1])
            cur = f if t == cur else t
        return path[::-1]

    def clear_cache(self):
        self._sp_cache.clear()
