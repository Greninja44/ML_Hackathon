
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import folium

TRIP_ID = "00000001"

trip_path = Path("data/raw/train") / TRIP_ID

track = pd.read_csv(
    trip_path / f"{TRIP_ID}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

# Basic GPS trajectory plot
plt.figure(figsize=(10, 7))

plt.plot(
    track["longitude"],
    track["latitude"],
    marker=".",
    markersize=2,
    linewidth=1
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"GPS Trajectory - Trip {TRIP_ID}")
plt.grid(True)

Path("visualizations").mkdir(exist_ok=True)

plt.savefig(
    f"visualizations/trip_{TRIP_ID}.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# Interactive Folium map
center = [
    track["latitude"].mean(),
    track["longitude"].mean()
]

m = folium.Map(location=center, zoom_start=14)

coordinates = track[
    ["latitude", "longitude"]
].values.tolist()

folium.PolyLine(
    coordinates,
    color="blue",
    weight=3,
    tooltip="GPS trajectory"
).add_to(m)

folium.Marker(
    coordinates[0],
    tooltip="Start",
    icon=folium.Icon(color="green")
).add_to(m)

folium.Marker(
    coordinates[-1],
    tooltip="End",
    icon=folium.Icon(color="red")
).add_to(m)

m.save(f"visualizations/trip_{TRIP_ID}.html")

print("GPS visualisations generated successfully!")
print("Observations:", len(track))
print("Map centre:", center)