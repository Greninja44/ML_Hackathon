"""Model features for (point, candidate) pairs from Person 1's CSV columns.

Only columns available at prediction time are used — never the route/labels.
Features that need geometry (projection position along the arc, node
out-degree, network distances) cannot be computed from the CSVs and are absent
here; they exist only in the raw-data pipeline (`src/data/features.py`).

=======================  ====================================================
distance_m               perpendicular point→arc distance (Person 1)
dist_rank                rank among the point's candidates (0 = nearest)
dist_minus_min           margin to the nearest candidate
n_candidates             1–5; fewer means sparse network or radius cut-off
direction_difference     |GPS bearing − arc bearing|, [0,180] (Person 1)
direction_axis           min(dd, 180−dd): parallel vs crossing, ignoring direction
heading_unreliable       point barely moved (gps_distance_m < 1 m): bearing meaningless
gps_speed_mps            speed from previous point (Person 1)
speed_spike              speed > 50 m/s (GPS jump; kept, flagged)
time_gap_sec             seconds since previous point
road_length_m            arc length (Person 1)
has_reverse_twin         the opposite-direction arc is also a candidate somewhere in the trip
ctx_mean_dist_m          mean distance of the ±3 neighbouring points to this arc
                         (lower bound = neighbour's farthest candidate if the arc is not its candidate)
ctx_hit_share            share of ±3 neighbours that also have this arc as a candidate
prev_link                2 = same arc as previous point's nearest candidate, 1 = follows it
                         (to_node == from_node), 0 = neither
next_link                same idea with the next point's nearest candidate
=======================  ====================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CSV_FEATURES = [
    "distance_m", "dist_rank", "dist_minus_min", "n_candidates", "direction_difference", "direction_axis",
    "heading_unreliable", "gps_speed_mps", "speed_spike", "time_gap_sec", "road_length_m", "has_reverse_twin",
    "ctx_mean_dist_m", "ctx_hit_share", "prev_link", "next_link",
]
CONTEXT = 3
SPIKE_MPS = 50.0


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """``pairs`` must be sorted by trip_id, gps_index, distance_m (load_partition does this)."""
    df = pairs.copy()
    key = ["trip_id", "gps_index"]
    g = df.groupby(key, sort=False)
    df["dist_rank"] = g.cumcount()
    df["dist_minus_min"] = df["distance_m"] - g["distance_m"].transform("min")
    df["n_candidates"] = g["distance_m"].transform("size")
    df["direction_axis"] = np.minimum(df["direction_difference"], 180.0 - df["direction_difference"])
    df["heading_unreliable"] = (df["gps_distance_m"] < 1.0).astype(int)
    df["speed_spike"] = (df["gps_speed_mps"] > SPIKE_MPS).astype(int)

    # reverse twin within the trip's candidate arcs
    arcs = df[["trip_id", "arc_id", "from_node_id", "to_node_id"]].drop_duplicates(["trip_id", "arc_id"])
    rev = arcs.rename(columns={"from_node_id": "to_node_id", "to_node_id": "from_node_id", "arc_id": "twin"})
    tw = arcs.merge(rev, on=["trip_id", "from_node_id", "to_node_id"], how="left")
    twin_set = set(zip(tw.loc[tw["twin"].notna(), "trip_id"], tw.loc[tw["twin"].notna(), "arc_id"]))
    df["has_reverse_twin"] = [int((t, a) in twin_set) for t, a in zip(df["trip_id"], df["arc_id"])]

    # neighbour context: distance of points i±o to the same arc
    far = g["distance_m"].transform("max")
    lookup = df[["trip_id", "gps_index", "arc_id", "distance_m"]]
    far_pt = df.drop_duplicates(key)[key].assign(far=g["distance_m"].max().to_numpy())
    sums = np.zeros(len(df)); hits = np.zeros(len(df)); cnt = np.zeros(len(df))
    for o in range(-CONTEXT, CONTEXT + 1):
        if o == 0:
            continue
        q = df[["trip_id", "gps_index", "arc_id"]].assign(gps_index=df["gps_index"] + o)
        m = q.merge(lookup, on=["trip_id", "gps_index", "arc_id"], how="left")
        f = q[key].merge(far_pt, on=key, how="left")["far"].to_numpy()
        exists = ~np.isnan(f)
        d = m["distance_m"].to_numpy()
        hit = ~np.isnan(d)
        val = np.where(hit, d, f)
        sums += np.where(exists, val, 0.0)
        cnt += exists
        hits += hit
    with np.errstate(invalid="ignore", divide="ignore"):
        df["ctx_mean_dist_m"] = np.where(cnt > 0, sums / cnt, df["distance_m"])
        df["ctx_hit_share"] = np.where(cnt > 0, hits / cnt, 0.0)
    del far

    # links with neighbouring points' nearest candidates (topology from node ids)
    nearest = df[df["dist_rank"] == 0][["trip_id", "gps_index", "arc_id", "from_node_id", "to_node_id"]]
    for name, shift in (("prev_link", 1), ("next_link", -1)):
        nb = nearest.assign(gps_index=nearest["gps_index"] + shift).rename(
            columns={"arc_id": "nb_arc", "from_node_id": "nb_from", "to_node_id": "nb_to"})
        m = df[key].merge(nb, on=key, how="left")
        same = m["nb_arc"].to_numpy() == df["arc_id"].to_numpy()
        if shift == 1:   # previous nearest arc → this arc
            follows = m["nb_to"].to_numpy() == df["from_node_id"].to_numpy()
        else:            # this arc → next nearest arc
            follows = df["to_node_id"].to_numpy() == m["nb_from"].to_numpy()
        df[name] = np.where(same, 2, np.where(follows, 1, 0))
    return df
