"""Per-trip data validation (Dataset Exploration §4–§6).

``validate_trip`` returns a flat dict of checks + a list of human-readable
issues. Nothing is modified; cleaning decisions live in ``preprocess.py``.
"""
from __future__ import annotations

import numpy as np

from src.data.io import TripData
from src.geo import LocalProjection, project_points_to_segments

LARGE_GAP_S = 10.0


def _bbox(lon, lat):
    return float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat))


def validate_trip(trip: TripData) -> dict:
    issues: list[str] = []
    tr = trip.track
    n = len(tr)
    out: dict = {"trip_id": trip.trip_id, "partition": trip.partition, "n_obs": n,
                 "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
                 "route_len": int(len(trip.route)) if trip.route is not None else 0,
                 "has_route": trip.has_route}
    for ext in (".track", ".nodes", ".arcs", ".route"):
        out[f"file{ext}"] = ext in trip.files
        out[f"bytes{ext}"] = trip.files[ext].stat().st_size if ext in trip.files else 0
    if not trip.has_route:
        issues.append("no reference route")
    if n == 0:
        issues.append("empty track")
        out["issues"] = issues
        return out

    vals = tr[["lon", "lat", "t"]].to_numpy(float)
    out["n_missing_values"] = int((~np.isfinite(vals)).sum())
    bad_range = (~tr["lon"].between(-180, 180)) | (~tr["lat"].between(-90, 90))
    out["n_out_of_range"] = int(bad_range.sum())
    if out["n_missing_values"]:
        issues.append(f"{out['n_missing_values']} missing/non-finite track values")
    if out["n_out_of_range"]:
        issues.append(f"{out['n_out_of_range']} coordinates out of lon/lat range")

    t = tr["t"].to_numpy(float)
    dt = np.diff(t)
    out["timestamps_monotonic"] = bool((dt >= 0).all()) if len(dt) else True
    out["n_duplicate_timestamps"] = int((dt == 0).sum())
    out["n_backward_timestamps"] = int((dt < 0).sum())
    out["share_dt_1s"] = float((dt == 1).mean()) if len(dt) else np.nan
    out["max_gap_s"] = float(dt.max()) if len(dt) else 0.0
    out["n_gaps_over_10s"] = int((dt > LARGE_GAP_S).sum())
    out["duration_s"] = float(t.max() - t.min()) if n else 0.0
    if not out["timestamps_monotonic"]:
        issues.append(f"{out['n_backward_timestamps']} backward time steps")
    if out["n_gaps_over_10s"]:
        issues.append(f"{out['n_gaps_over_10s']} GPS gaps > {LARGE_GAP_S:.0f}s (max {out['max_gap_s']:.0f}s)")

    lon, lat = tr["lon"].to_numpy(float), tr["lat"].to_numpy(float)
    out.update(dict(zip(["min_lon", "min_lat", "max_lon", "max_lat"], _bbox(lon, lat))))
    # coordinate-order plausibility: GPS should overlap the node bounding box
    if len(trip.nodes):
        nb = _bbox(trip.nodes[:, 0], trip.nodes[:, 1])
        inside = ((lon >= nb[0] - 0.01) & (lon <= nb[2] + 0.01) & (lat >= nb[1] - 0.01) & (lat <= nb[3] + 0.01))
        out["share_gps_inside_network_bbox"] = float(np.nanmean(inside))
        swapped = ((lat >= nb[0] - 0.01) & (lat <= nb[2] + 0.01) & (lon >= nb[1] - 0.01) & (lon <= nb[3] + 0.01))
        if np.nanmean(inside) < 0.5 and np.nanmean(swapped) > 0.5:
            issues.append("GPS coordinates appear to be in lat/lon order rather than lon/lat")

    # network integrity
    arcs = trip.arcs
    nn = len(trip.nodes)
    bad_arc = ~((arcs >= 0) & (arcs < nn)).all(axis=1) if len(arcs) else np.zeros(0, bool)
    out["n_arcs_bad_node_ref"] = int(bad_arc.sum())
    out["n_self_loops"] = int((arcs[:, 0] == arcs[:, 1]).sum()) if len(arcs) else 0
    out["n_duplicate_arcs"] = int(len(arcs) - len({tuple(a) for a in arcs})) if len(arcs) else 0
    if len(arcs):
        pairs = {tuple(a) for a in arcs}
        out["share_arcs_with_reverse"] = float(np.mean([(b, a) in pairs for a, b in arcs]))
        used = np.zeros(nn, bool)
        good = arcs[~bad_arc]
        used[good.ravel()] = True
        out["n_isolated_nodes"] = int((~used).sum())
    if out["n_arcs_bad_node_ref"]:
        issues.append(f"{out['n_arcs_bad_node_ref']} arcs reference missing nodes")

    # reference route integrity
    if trip.has_route:
        r = trip.route
        bad = (r < 0) | (r >= len(arcs))
        out["n_route_bad_ids"] = int(bad.sum())
        rv = r[~bad]
        if len(rv) > 1:
            connected = arcs[rv[:-1], 1] == arcs[rv[1:], 0]
            repeat = rv[:-1] == rv[1:]
            out["route_share_connected"] = float((connected | repeat).mean())
            out["route_n_breaks"] = int((~(connected | repeat)).sum())
            if out["route_n_breaks"]:
                issues.append(f"reference route has {out['route_n_breaks']} non-adjacent consecutive arcs")
        if out["n_route_bad_ids"]:
            issues.append(f"{out['n_route_bad_ids']} route arc ids out of range")

    # how far does the GPS lie from the reference route? (label plausibility)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if trip.has_route and ok.sum() and len(trip.nodes) and out.get("n_route_bad_ids", 0) < len(trip.route):
        proj = LocalProjection.for_lonlat(lon[ok], lat[ok])
        xy = proj.to_xy(lon[ok], lat[ok])
        nxy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        rv = np.unique(trip.route[(trip.route >= 0) & (trip.route < len(arcs))])
        rv = rv[~bad_arc[rv]]
        if len(rv):
            a, b = nxy[arcs[rv, 0]], nxy[arcs[rv, 1]]
            step = max(1, len(xy) // 2000)
            d = np.array([project_points_to_segments(np.repeat(p[None], len(rv), 0), a, b)[0].min()
                          for p in xy[::step]])
            out["gps_to_route_median_m"] = float(np.median(d))
            out["gps_to_route_p95_m"] = float(np.percentile(d, 95))
            if out["gps_to_route_p95_m"] > 50:
                issues.append(f"5% of GPS points are > {out['gps_to_route_p95_m']:.0f} m from reference route")
    out["issues"] = issues
    out["n_issues"] = len(issues)
    return out
