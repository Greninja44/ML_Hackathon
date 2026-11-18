
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

TRIP_ID = "00000001"

BASE = Path("data/raw/train") / TRIP_ID

# Load input data
track = pd.read_csv(
    BASE / f"{TRIP_ID}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

nodes = pd.read_csv(
    BASE / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

arcs = pd.read_csv(
    BASE / f"{TRIP_ID}.arcs",
    sep=r"\s+",
    header=None,
    names=["from_node_id", "to_node_id"]
)

candidates = pd.read_csv(
    f"data/processed/candidates_{TRIP_ID}.csv"
)

# Convert coordinates to metres
mean_lon = nodes["longitude"].mean()
mean_lat = nodes["latitude"].mean()

zone = int((mean_lon + 180) / 6) + 1
epsg = (32600 if mean_lat >= 0 else 32700) + zone

transformer = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg}",
    always_xy=True
)

gps_x, gps_y = transformer.transform(
    track["longitude"].to_numpy(),
    track["latitude"].to_numpy()
)

node_x, node_y = transformer.transform(
    nodes["longitude"].to_numpy(),
    nodes["latitude"].to_numpy()
)

gps = np.column_stack((gps_x, gps_y))
node_coords = np.column_stack((node_x, node_y))

# GPS movement features
gps_distance = np.zeros(len(track))
gps_bearing = np.zeros(len(track))
time_gap = np.zeros(len(track))

for i in range(1, len(track)):
    dx = gps[i, 0] - gps[i-1, 0]
    dy = gps[i, 1] - gps[i-1, 1]

    gps_distance[i] = np.hypot(dx, dy)

    gps_bearing[i] = (
        np.degrees(np.arctan2(dx, dy)) + 360
    ) % 360

    time_gap[i] = (
        track.iloc[i]["timestamp"]
        - track.iloc[i-1]["timestamp"]
    )

gps_speed = np.divide(
    gps_distance,
    time_gap,
    out=np.zeros_like(gps_distance),
    where=time_gap > 0
)

# Road segment features
edges = arcs[
    ["from_node_id", "to_node_id"]
].to_numpy(dtype=int)

road_features = []

for arc_id, (start_id, end_id) in enumerate(edges):

    start = node_coords[start_id]
    end = node_coords[end_id]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = np.hypot(dx, dy)

    bearing = (
        np.degrees(np.arctan2(dx, dy)) + 360
    ) % 360

    road_features.append({
        "arc_id": arc_id,
        "road_length_m": length,
        "road_bearing": bearing
    })

road_df = pd.DataFrame(road_features)

# Combine candidate records with road features
features = candidates.merge(
    road_df,
    on="arc_id",
    how="left"
)

features["gps_distance_m"] = (
    features["gps_index"].map(
        dict(enumerate(gps_distance))
    )
)

features["gps_bearing"] = (
    features["gps_index"].map(
        dict(enumerate(gps_bearing))
    )
)

features["time_gap_sec"] = (
    features["gps_index"].map(
        dict(enumerate(time_gap))
    )
)

features["gps_speed_mps"] = (
    features["gps_index"].map(
        dict(enumerate(gps_speed))
    )
)

# Direction difference, accounting for undirected orientation
difference = abs(
    features["gps_bearing"] - features["road_bearing"]
)

features["direction_difference"] = np.minimum(
    difference,
    360 - difference
)

# Save features
Path("data/processed").mkdir(exist_ok=True)

output = f"data/processed/features_{TRIP_ID}.csv"

features.to_csv(output, index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("Total candidate rows:", len(features))
print("\nGenerated features:")
print(features.columns.tolist())
print("\nSaved to:", output)