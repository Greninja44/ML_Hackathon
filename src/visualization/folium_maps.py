"""Folium maps for EDA and the report (Deliverables §6).

Layers (each toggleable via LayerControl): relevant road network, raw GPS
observations, reference route (offline/report use only — never served to the
UI), and one layer per predicted route. Includes a legend, start/end markers
and fits the view to the trajectory.
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np

from src.data.network import RoadNetwork

COLORS = {
    "network": "#9a9890",
    "gps": "#2a78d6",
    "reference": "#008300",
    "pred": ["#eb6834", "#6250d6", "#e34948", "#eda100"],
}


def _arc_lines(net: RoadNetwork, arc_ids) -> list[list[tuple[float, float]]]:
    lines = []
    for a in arc_ids:
        if 0 <= a < net.n_arcs and net.valid[a]:
            f, t = net.arcs[a]
            lines.append([(net.node_lonlat[f, 1], net.node_lonlat[f, 0]),
                          (net.node_lonlat[t, 1], net.node_lonlat[t, 0])])
    return lines


def _route_polyline(net: RoadNetwork, arc_ids) -> list[list[tuple[float, float]]]:
    """Join consecutive connected arcs into continuous polylines."""
    out, cur, last_to = [], [], None
    for a in arc_ids:
        if not (0 <= a < net.n_arcs and net.valid[a]):
            continue
        f, t = net.arcs[a]
        pf = (net.node_lonlat[f, 1], net.node_lonlat[f, 0])
        pt = (net.node_lonlat[t, 1], net.node_lonlat[t, 0])
        if cur and last_to == f:
            cur.append(pt)
        else:
            if cur:
                out.append(cur)
            cur = [pf, pt]
        last_to = t
    if cur:
        out.append(cur)
    return out


def _legend(items: list[tuple[str, str]]) -> folium.Element:
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="width:18px;height:4px;background:{c};display:inline-block;border-radius:2px"></span>{n}</div>'
        for n, c in items)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #ccc;border-radius:6px;font:12px sans-serif;color:#222">{rows}</div>')
    return folium.Element(html)


def trip_map(net: RoadNetwork, gps_lonlat: np.ndarray, reference_arcs=None, predictions: dict | None = None,
             network_buffer_m: float = 150.0, title: str | None = None) -> folium.Map:
    gps = np.asarray(gps_lonlat, float)
    gps = gps[np.isfinite(gps).all(axis=1)]
    center = [float(np.median(gps[:, 1])), float(np.median(gps[:, 0]))] if len(gps) else [0, 0]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    legend = []

    # road network near the trajectory only (relevant geometry)
    if len(gps):
        xy = net.proj.to_xy(gps[:, 0], gps[:, 1])
        near = set()
        for p in xy[:: max(1, len(xy) // 400)]:
            near.update(net.arcs_near(p, network_buffer_m)[0].tolist())
        fg = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
        for line in _arc_lines(net, sorted(near)):
            folium.PolyLine(line, color=COLORS["network"], weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))

    if reference_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in _route_polyline(net, reference_arcs):
            folium.PolyLine(line, color=COLORS["reference"], weight=6, opacity=0.7).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    for k, (name, arcs) in enumerate((predictions or {}).items()):
        color = COLORS["pred"][k % len(COLORS["pred"])]
        fg = folium.FeatureGroup(name=f"Predicted: {name}", show=True)
        for line in _route_polyline(net, arcs):
            folium.PolyLine(line, color=color, weight=4, opacity=0.9, dash_array="8 6" if k else None,
                            tooltip=name).add_to(fg)
        fg.add_to(m)
        legend.append((f"Predicted: {name}", color))

    fg = folium.FeatureGroup(name="GPS observations", show=True)
    for i, (lon, lat) in enumerate(gps):
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"obs {i}").add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))
    if len(gps):
        folium.Marker((gps[0, 1], gps[0, 0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker((gps[-1, 1], gps[-1, 0]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
        m.fit_bounds([[gps[:, 1].min(), gps[:, 0].min()], [gps[:, 1].max(), gps[:, 0].max()]])
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:4px 8px;'
            f'border-radius:4px;font:bold 13px sans-serif">{title}</div>'))
    m.get_root().html.add_child(_legend(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_map(m: folium.Map, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path
