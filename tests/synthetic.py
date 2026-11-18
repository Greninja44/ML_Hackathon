"""SYNTHETIC test fixture — NOT project data.

Generates small trips in the documented raw format (Dataset Exploration §3) on
a regular grid road network so the pipeline can be unit- and smoke-tested
before / without the real dataset. Nothing produced from this module may be
reported as an experimental result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ORIGIN_LON, ORIGIN_LAT = 4.90, 52.37  # arbitrary plausible origin
M_PER_DEG_LAT = 111_320.0


def _to_lonlat(x, y):
    lon = ORIGIN_LON + x / (M_PER_DEG_LAT * np.cos(np.radians(ORIGIN_LAT)))
    lat = ORIGIN_LAT + y / M_PER_DEG_LAT
    return lon, lat


def make_grid(n: int = 6, spacing: float = 120.0):
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    arcs = []
    for r in range(n):
        for c in range(n):
            u = r * n + c
            if c + 1 < n:
                arcs += [(u, u + 1), (u + 1, u)]
            if r + 1 < n:
                arcs += [(u, u + n), (u + n, u)]
    return xy, np.array(arcs)


def random_route(xy, arcs, n_arcs: int, rng) -> list[int]:
    out_arcs: dict[int, list[int]] = {}
    for i, (f, _t) in enumerate(arcs):
        out_arcs.setdefault(int(f), []).append(i)
    cur = int(rng.integers(len(arcs)))
    route = [cur]
    for _ in range(n_arcs - 1):
        f, t = arcs[cur]
        nxt = [a for a in out_arcs[int(t)] if arcs[a, 1] != f]  # no U-turns
        cur = int(rng.choice(nxt))
        route.append(cur)
    return route


def simulate_track(xy, arcs, route, rng, speed=9.0, noise=5.0, dt=1.0):
    pts = []
    carry = 0.0
    for a in route:
        p0, p1 = xy[arcs[a, 0]], xy[arcs[a, 1]]
        L = float(np.hypot(*(p1 - p0)))
        s = carry
        while s < L:
            pts.append(p0 + (p1 - p0) * (s / L))
            s += speed * dt
        carry = s - L
    pts = np.array(pts) + rng.normal(0, noise, (len(pts), 2))
    t = 1_700_000_000 + np.arange(len(pts)) * dt
    return pts, t


def write_trip(directory: Path, trip_id: str, seed: int, n_route_arcs: int = 12, with_route=True):
    rng = np.random.default_rng(seed)
    xy, arcs = make_grid()
    route = random_route(xy, arcs, n_route_arcs, rng)
    pts, t = simulate_track(xy, arcs, route, rng)
    d = Path(directory) / trip_id
    d.mkdir(parents=True, exist_ok=True)
    lon, lat = _to_lonlat(pts[:, 0], pts[:, 1])
    np.savetxt(d / f"{trip_id}.track", np.column_stack([lon, lat, t]), fmt=["%.7f", "%.7f", "%.0f"])
    nlon, nlat = _to_lonlat(xy[:, 0], xy[:, 1])
    np.savetxt(d / f"{trip_id}.nodes", np.column_stack([nlon, nlat]), fmt="%.7f")
    np.savetxt(d / f"{trip_id}.arcs", arcs, fmt="%d")
    if with_route:
        np.savetxt(d / f"{trip_id}.route", np.array(route), fmt="%d")
    return route


def write_dataset(root: Path, n_train=8, n_student=3, n_eval=2, seed=0) -> Path:
    root = Path(root)
    k = seed
    for part, n in (("train", n_train), ("student_test", n_student), ("eval_test", n_eval)):
        for i in range(n):
            write_trip(root / part, f"syn_{part}_{i:02d}", seed=k)
            k += 1
    return root
