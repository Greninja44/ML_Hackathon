
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TRIP_ID = "00000001"

trip_path = Path("data/raw/train") / TRIP_ID

nodes = pd.read_csv(
    trip_path / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

arcs = pd.read_csv(
    trip_path / f"{TRIP_ID}.arcs",
    sep=r"\s+",
    header=None,
    names=["from_node_id", "to_node_id"]
)

print("Nodes:", len(nodes))
print("Road connections:", len(arcs))

# Convert node coordinates to a NumPy array
coordinates = nodes[["longitude", "latitude"]].to_numpy()

# Build all road segments at once
edges = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)

segments = coordinates[edges]

print("Preparing road network plot...")

fig, ax = plt.subplots(figsize=(12, 8))

collection = LineCollection(
    segments,
    colors="gray",
    linewidths=0.35
)

ax.add_collection(collection)

ax.autoscale()
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Road Network - Trip {TRIP_ID}")
ax.grid(True)

Path("visualizations").mkdir(exist_ok=True)

plt.savefig(
    f"visualizations/road_network_{TRIP_ID}.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("Road network visualisation saved successfully!")