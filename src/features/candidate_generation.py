
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer

TRIP_ID = "00000001"
K = 20
MAX_CANDIDATES = 5
MAX_DISTANCE = 50.0

trip_path = Path("data/raw/train") / TRIP_ID

# Load GPS observations
track = pd.read_csv(
    trip_path / f"{TRIP_ID}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

# Load road nodes
nodes = pd.read_csv(
    trip_path / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

# Load road connections
arcs = pd.read_csv(
    trip_path / f"{TRIP_ID}.arcs",
    sep=r"\s+",
    header=None,
    names=["from_node_id", "to_node_id"]
)

# Convert geographic coordinates to metres
mean_lon = nodes["longitude"].mean()
mean_lat = nodes["latitude"].mean()

zone = int((mean_lon + 180) / 6) + 1
epsg = (32600 if mean_lat >= 0 else 32700) + zone

transformer = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg}",
    always_xy=True
)

node_x, node_y = transformer.transform(
    nodes["longitude"].to_numpy(),
    nodes["latitude"].to_numpy()
)

gps_x, gps_y = transformer.transform(
    track["longitude"].to_numpy(),
    track["latitude"].to_numpy()
)

node_coords = np.column_stack((node_x, node_y))
gps_coords = np.column_stack((gps_x, gps_y))

# Build road segments using arc IDs
edges = arcs[
    ["from_node_id", "to_node_id"]
].to_numpy(dtype=int)

segments = node_coords[edges]

# Build spatial index using segment midpoints
midpoints = segments.mean(axis=1)
tree = cKDTree(midpoints)

candidate_rows = []

print("Generating road candidates...")

for gps_index, point in enumerate(gps_coords):

    k = min(K, len(segments))

    # Find nearby segment midpoints
    _, indices = tree.query(point, k=k)
    indices = np.atleast_1d(indices)

    possible_segments = segments[indices]

    # Calculate exact point-to-segment distances
    start = possible_segments[:, 0, :]
    end = possible_segments[:, 1, :]

    direction = end - start
    length_squared = np.sum(direction ** 2, axis=1)

    length_squared = np.maximum(length_squared, 1e-12)

    t = np.sum(
        (point - start) * direction,
        axis=1
    ) / length_squared

    t = np.clip(t, 0, 1)

    closest = start + t[:, None] * direction

    distances = np.linalg.norm(
        point - closest,
        axis=1
    )

    # Sort candidates by exact distance
    order = np.argsort(distances)

    selected = [
        i for i in order
        if distances[i] <= MAX_DISTANCE
    ][:MAX_CANDIDATES]

    # Keep nearest candidate if none is within radius
    if not selected:
        selected = [order[0]]

    for position in selected:
        arc_id = int(indices[position])

        candidate_rows.append({
            "gps_index": gps_index,
            "timestamp": track.iloc[gps_index]["timestamp"],
            "arc_id": arc_id,
            "from_node_id": int(edges[arc_id, 0]),
            "to_node_id": int(edges[arc_id, 1]),
            "distance_m": float(distances[position])
        })

# Save results
result = pd.DataFrame(candidate_rows)

Path("data/processed").mkdir(exist_ok=True)

output_path = (
    Path("data/processed")
    / f"candidates_{TRIP_ID}.csv"
)

result.to_csv(output_path, index=False)

print("\nCandidate generation completed!")
print("GPS observations:", len(track))
print("Candidate records:", len(result))
print("Average candidates per GPS point:",
      round(len(result) / len(track), 2))
print("Saved to:", output_path)