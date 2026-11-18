
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TRIP_ID = "00000001"

trip_path = Path("data/raw/train") / TRIP_ID

# Load GPS observations
track = pd.read_csv(
    trip_path / f"{TRIP_ID}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

# Load road network
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

# Create road segments
coordinates = nodes[["longitude", "latitude"]].to_numpy()
edges = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
segments = coordinates[edges]

# Plot
fig, ax = plt.subplots(figsize=(13, 9))

road_lines = LineCollection(
    segments,
    colors="gray",
    linewidths=0.3,
    alpha=0.5
)

ax.add_collection(road_lines)

# Overlay GPS trajectory
ax.plot(
    track["longitude"],
    track["latitude"],
    color="red",
    linewidth=1.5,
    label="GPS trajectory",
    zorder=3
)

ax.scatter(
    track["longitude"],
    track["latitude"],
    color="red",
    s=2,
    zorder=4
)

ax.set_xlim(
    track["longitude"].min() - 0.003,
    track["longitude"].max() + 0.003
)

ax.set_ylim(
    track["latitude"].min() - 0.003,
    track["latitude"].max() + 0.003
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("GPS Trajectory vs Road Network")
ax.legend()
ax.grid(True)

Path("visualizations").mkdir(exist_ok=True)

plt.savefig(
    "visualizations/gps_road_overlay.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print("GPS and road network overlay generated!")