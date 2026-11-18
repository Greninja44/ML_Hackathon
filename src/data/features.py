"""Feature engineering for (observation, candidate-arc) pairs.

Every feature is computed from the GPS track and road network only — never
from the reference route — so the same function is used for training, offline
evaluation and the live API (no train/serve skew, no label leakage).

Feature rationale (see docs/METHODOLOGY.md for the full discussion):

=====================  =====================================================
dist_m                 perpendicular GPS→arc distance; the classic emission cue
dist_rank              rank of the candidate among this observation's candidates
dist_minus_min         margin to the nearest candidate (ambiguity measure)
n_candidates           local road density / ambiguity
used_fallback          observation needed the widened search radius
heading_diff           GPS movement bearing vs. *directed* arc bearing, [0,180]
heading_diff_axis      same but ignoring direction, [0,90] (parallel vs crossing)
heading_missing        vehicle (nearly) stationary, heading undefined
speed_mps              centred speed estimate
arc_len_m              segment length
t_along                where on the arc the point projects (0=start, 1=end)
endpoint_dist_m        distance along arc to the nearest endpoint (junction proximity)
has_reverse_twin       a reverse arc exists (two-way road stored as two arcs)
ctx_mean_dist_m        mean distance of neighbouring observations to this arc
ctx_max_dist_m         max distance of neighbouring observations to this arc
prev_link              candidate equals / follows the previous observation's nearest arc
out_degree             branching factor at the arc's end node
dt_prev_s              time gap since previous observation
outlier                implausible-speed flag from preprocessing
=====================  =====================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.network import RoadNetwork
from src.data.preprocess import CleanTrack
from src.geo import angle_diff_deg, project_points_to_segments
from src.map_matching.candidates import CandidateSet

FEATURE_COLUMNS = [
    "dist_m", "dist_rank", "dist_minus_min", "n_candidates", "used_fallback",
    "heading_diff", "heading_diff_axis", "heading_missing", "speed_mps",
    "arc_len_m", "t_along", "endpoint_dist_m", "has_reverse_twin",
    "ctx_mean_dist_m", "ctx_max_dist_m", "prev_link", "out_degree",
    "dt_prev_s", "outlier",
]
CONTEXT_WINDOW = 3


def build_features(track: CleanTrack, cands: CandidateSet, net: RoadNetwork) -> pd.DataFrame:
    P = len(cands.arc)
    if P == 0:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURE_COLUMNS)
    obs, arc = cands.obs, cands.arc
    counts = cands.counts()

    # rank & margin within each observation (pairs are sorted by distance per obs)
    rank = np.arange(P) - cands.offsets[obs]
    first = cands.offsets[obs]
    dist_min = cands.dist[first]

    gps_heading = track.heading_deg[obs]
    missing = ~np.isfinite(gps_heading)
    hd = angle_diff_deg(np.nan_to_num(gps_heading), net.arc_bearing[arc])
    hd = np.where(missing, 90.0, hd)
    hd_axis = np.minimum(hd, 180.0 - hd)

    n_obs = len(track)
    ctx = []
    for o in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
        if o == 0:
            continue
        j = np.clip(obs + o, 0, n_obs - 1)
        d, _ = project_points_to_segments(track.xy[j], net.arc_a[arc], net.arc_b[arc])
        ctx.append(d)
    ctx = np.vstack(ctx + [cands.dist])

    # previous observation's nearest arc (GPS-only information)
    nearest = np.full(n_obs, -1)
    has = counts > 0
    nearest[has] = cands.arc[cands.offsets[:-1][has]]
    prev_near = np.where(obs > 0, nearest[np.maximum(obs - 1, 0)], -1)
    prev_ok = prev_near >= 0
    same = prev_ok & (prev_near == arc)
    follows = prev_ok & (net.arcs[np.maximum(prev_near, 0), 1] == net.arcs[arc, 0])
    if not net.directed:
        pa = net.arcs[np.maximum(prev_near, 0)]
        ca = net.arcs[arc]
        shares_node = ((pa[:, 0] == ca[:, 0]) | (pa[:, 0] == ca[:, 1])
                       | (pa[:, 1] == ca[:, 0]) | (pa[:, 1] == ca[:, 1]))
        follows |= prev_ok & shares_node
    prev_link = np.where(same, 2, np.where(follows, 1, 0))

    out_deg = np.array([len(net.out_arcs[int(n)]) for n in net.arcs[arc, 1]])
    twin = np.array([net.reverse_arc(int(a)) is not None for a in arc])
    arc_len = net.arc_len[arc]

    df = pd.DataFrame({
        "obs": obs,
        "arc": arc,
        "dist_m": cands.dist,
        "dist_rank": rank,
        "dist_minus_min": cands.dist - dist_min,
        "n_candidates": counts[obs],
        "used_fallback": cands.fallback.astype(int),
        "heading_diff": hd,
        "heading_diff_axis": hd_axis,
        "heading_missing": missing.astype(int),
        "speed_mps": np.nan_to_num(track.speed_mps[obs], nan=0.0),
        "arc_len_m": arc_len,
        "t_along": cands.t_along,
        "endpoint_dist_m": np.minimum(cands.t_along, 1 - cands.t_along) * arc_len,
        "has_reverse_twin": twin.astype(int),
        "ctx_mean_dist_m": ctx.mean(axis=0),
        "ctx_max_dist_m": ctx.max(axis=0),
        "prev_link": prev_link,
        "out_degree": out_deg,
        "dt_prev_s": np.nan_to_num(track.dt[obs], nan=0.0),
        "outlier": track.outlier[obs].astype(int),
    })
    return df
