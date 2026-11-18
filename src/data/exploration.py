"""Dataset exploration (Dataset Exploration §4 and §8).

Development partitions only — the evaluator partition is not opened.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DEVELOPMENT_PARTITIONS  # noqa: E402
from src.data.io import TripData, iter_partition, list_partition  # noqa: E402
from src.data.validation import validate_trip  # noqa: E402
from src.geo import haversine_m  # noqa: E402
from src.visualization.plots import INK_2, SERIES, _save, _style  # noqa: E402


def inventory(cfg: dict) -> pd.DataFrame:
    rows = []
    for part in DEVELOPMENT_PARTITIONS:
        for tid, files in list_partition(cfg, part).items():
            rows.append({"partition": part, "trip_id": tid,
                         **{f"has{e}": e in files for e in (".track", ".nodes", ".arcs", ".route")}})
    return pd.DataFrame(rows)


def network_stats(trip: TripData) -> dict:
    g = nx.DiGraph()
    g.add_nodes_from(range(len(trip.nodes)))
    ok = ((trip.arcs >= 0) & (trip.arcs < len(trip.nodes))).all(axis=1)
    g.add_edges_from(map(tuple, trip.arcs[ok]))
    weak = list(nx.weakly_connected_components(g))
    strong = list(nx.strongly_connected_components(g))
    deg = np.array([d for _, d in g.degree()])
    return {
        "trip_id": trip.trip_id,
        "n_weak_components": len(weak),
        "largest_weak_component_share": max(map(len, weak)) / max(g.number_of_nodes(), 1) if weak else 0,
        "n_strong_components": len(strong),
        "largest_strong_component_share": max(map(len, strong)) / max(g.number_of_nodes(), 1) if strong else 0,
        "n_isolated_nodes": int((deg == 0).sum()),
        "mean_degree": float(deg.mean()) if len(deg) else 0.0,
    }


def trip_summary(trip: TripData) -> dict:
    tr = trip.track.dropna()
    d = haversine_m(tr["lon"].values[:-1], tr["lat"].values[:-1], tr["lon"].values[1:], tr["lat"].values[1:])
    dt = np.diff(tr["t"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, d / dt, np.nan)
    return {"trip_id": trip.trip_id, "length_km": float(d.sum() / 1000),
            "median_step_m": float(np.median(d)) if len(d) else np.nan,
            "median_speed_kmh": float(np.nanmedian(speed) * 3.6) if len(d) else np.nan,
            "p99_speed_kmh": float(np.nanpercentile(speed, 99) * 3.6) if len(d) else np.nan}


def explore(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"inventory": inventory(cfg)}
    val, net, summ, dts = [], [], [], []
    for part in DEVELOPMENT_PARTITIONS:
        for trip in iter_partition(cfg, part):
            v = validate_trip(trip)
            v["issues"] = "; ".join(v["issues"])
            val.append(v)
            net.append({"partition": part, **network_stats(trip)})
            summ.append({"partition": part, **trip_summary(trip)})
            dts.append(pd.DataFrame({"partition": part, "dt": np.diff(trip.track["t"].to_numpy(float))}))
    tables["validation"] = pd.DataFrame(val)
    tables["network"] = pd.DataFrame(net)
    tables["trip_summary"] = pd.DataFrame(summ)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    if val:
        v = tables["validation"]
        tables["partition_summary"] = v.groupby("partition").agg(
            n_trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
            obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), nodes_median=("n_nodes", "median"),
            arcs_median=("n_arcs", "median"), missing_values=("n_missing_values", "sum"),
            trips_with_gaps=("n_gaps_over_10s", lambda s: int((s > 0).sum())),
            trips_with_issues=("n_issues", lambda s: int((s > 0).sum()))).reset_index()
        tables["partition_summary"].to_csv(out_dir / "partition_summary.csv", index=False)
        _plots(tables, pd.concat(dts, ignore_index=True) if dts else pd.DataFrame(), out_dir)
    return tables


def _hist(values, path, title, xlabel, bins=40, log=False):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(values, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.8, log=log)
    _style(ax, title, xlabel, "count")
    ax.grid(axis="y", color="#e4e3df")
    _save(fig, path)


def _plots(tables, dts: pd.DataFrame, out: Path):
    v, s = tables["validation"], tables["trip_summary"]
    _hist(v["n_obs"], out / "obs_per_trip.png", "GPS observations per trip (train + student-test)", "observations")
    _hist(s["length_km"], out / "trip_length_km.png", "Trip length", "km")
    if len(dts):
        clipped = dts["dt"].clip(upper=30)
        _hist(clipped, out / "sampling_interval.png", "Sampling interval (clipped at 30 s)", "seconds",
              bins=60, log=True)
    if "gps_to_route_median_m" in v:
        _hist(v["gps_to_route_median_m"].dropna(), out / "gps_to_route_distance.png",
              "Median GPS-to-reference-route distance per trip", "metres")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(v["n_arcs"], v["n_obs"], s=14, color=SERIES[0])
    _style(ax, "Network size vs trajectory length per trip", "arcs in trip network", "GPS observations")
    _save(fig, out / "network_vs_track.png")
    missing = v[["trip_id", "n_missing_values", "n_out_of_range", "n_duplicate_timestamps",
                 "n_backward_timestamps", "n_gaps_over_10s"]].set_index("trip_id")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    totals = missing.sum()
    ax.barh(totals.index, totals.values, color=SERIES[0], height=0.6)
    for y, val in enumerate(totals.values):
        ax.text(val, y, f"  {int(val)}", va="center", color=INK_2, fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Data-quality counts across all development trips")
    _save(fig, out / "missing_and_anomalies.png")


def plot_trip(trip: TripData, path: Path):
    """Static trajectory + network plot in lon/lat (for the report)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for f, t in trip.arcs:
        if 0 <= f < len(trip.nodes) and 0 <= t < len(trip.nodes):
            ax.plot(trip.nodes[[f, t], 0], trip.nodes[[f, t], 1], color="#c9c7bf", linewidth=0.6)
    ax.scatter(trip.track["lon"], trip.track["lat"], s=3, color=SERIES[0], label="GPS observations")
    if trip.has_route:
        for a in trip.route:
            if 0 <= a < len(trip.arcs):
                f, t = trip.arcs[a]
                ax.plot(trip.nodes[[f, t], 0], trip.nodes[[f, t], 1], color="#008300", linewidth=2)
    ax.set_aspect(1 / np.cos(np.radians(trip.track["lat"].median())))
    _style(ax, f"Trip {trip.trip_id}: GPS (blue), reference route (green), network (grey)", "longitude", "latitude")
    _save(fig, path)
