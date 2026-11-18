
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer

RAW = Path("data/raw")
OUT = Path("data/processed")

OUT.mkdir(parents=True, exist_ok=True)

K = 20
MAX_CANDIDATES = 5
MAX_DISTANCE = 50.0


def process_trip(trip_path, partition):
    trip_id = trip_path.name

    track = pd.read_csv(
        trip_path / f"{trip_id}.track",
        sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_path / f"{trip_id}.nodes",
        sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_path / f"{trip_id}.arcs",
        sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"]
    )

    # Coordinate projection
    lon, lat = nodes["longitude"].mean(), nodes["latitude"].mean()
    zone = int((lon + 180) / 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}",
        always_xy=True
    )

    nx, ny = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )

    gx, gy = transformer.transform(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy()
    )

    node_xy = np.column_stack((nx, ny))
    gps_xy = np.column_stack((gx, gy))

    edges = arcs.to_numpy(dtype=int)
    segments = node_xy[edges]

    # Candidate generation
    tree = cKDTree(segments.mean(axis=1))
    rows = []

    for i, point in enumerate(gps_xy):
        k = min(K, len(segments))
        _, indices = tree.query(point, k=k)
        indices = np.atleast_1d(indices)

        selected_segments = segments[indices]
        start = selected_segments[:, 0]
        direction = selected_segments[:, 1] - start

        denom = np.maximum(
            np.sum(direction ** 2, axis=1), 1e-12
        )

        t = np.clip(
            np.sum((point - start) * direction, axis=1)
            / denom, 0, 1
        )

        closest = start + t[:, None] * direction
        distances = np.linalg.norm(point - closest, axis=1)

        order = np.argsort(distances)
        selected = [
            j for j in order
            if distances[j] <= MAX_DISTANCE
        ][:MAX_CANDIDATES]

        if not selected:
            selected = [order[0]]

        for j in selected:
            arc_id = int(indices[j])
            rows.append({
                "partition": partition,
                "trip_id": trip_id,
                "gps_index": i,
                "timestamp": track.iloc[i]["timestamp"],
                "arc_id": arc_id,
                "from_node_id": int(edges[arc_id, 0]),
                "to_node_id": int(edges[arc_id, 1]),
                "distance_m": float(distances[j])
            })

    result = pd.DataFrame(rows)

    # GPS movement features
    dx = np.diff(gps_xy[:, 0], prepend=gps_xy[0, 0])
    dy = np.diff(gps_xy[:, 1], prepend=gps_xy[0, 1])

    gps_distance = np.hypot(dx, dy)

    gps_bearing = (
        np.degrees(np.arctan2(dx, dy)) + 360
    ) % 360

    time_gap = track["timestamp"].diff().fillna(0).to_numpy()

    gps_speed = np.divide(
        gps_distance, time_gap,
        out=np.zeros_like(gps_distance),
        where=time_gap > 0
    )

    # Road features
    road_dx = segments[:, 1, 0] - segments[:, 0, 0]
    road_dy = segments[:, 1, 1] - segments[:, 0, 1]

    road_length = np.hypot(road_dx, road_dy)

    road_bearing = (
        np.degrees(np.arctan2(road_dx, road_dy)) + 360
    ) % 360

    result["gps_distance_m"] = result["gps_index"].map(
        dict(enumerate(gps_distance))
    )
    result["gps_bearing"] = result["gps_index"].map(
        dict(enumerate(gps_bearing))
    )
    result["time_gap_sec"] = result["gps_index"].map(
        dict(enumerate(time_gap))
    )
    result["gps_speed_mps"] = result["gps_index"].map(
        dict(enumerate(gps_speed))
    )

    result["road_length_m"] = result["arc_id"].map(
        dict(enumerate(road_length))
    )
    result["road_bearing"] = result["arc_id"].map(
        dict(enumerate(road_bearing))
    )

    difference = abs(
        result["gps_bearing"] - result["road_bearing"]
    )

    result["direction_difference"] = np.minimum(
        difference, 360 - difference
    )

    # Reference labels kept separately, never used as features
    route_file = trip_path / f"{trip_id}.route"

    labels = pd.read_csv(
        route_file, sep=r"\s+",
        header=None, names=["arc_id"]
    )

    label_df = pd.DataFrame({
        "partition": partition,
        "trip_id": trip_id,
        "route_position": np.arange(len(labels)),
        "true_arc_id": labels["arc_id"]
    })

    return result, label_df


def main():
    for partition in ["train", "student_test"]:
        all_features = []
        all_labels = []

        folders = sorted((RAW / partition).iterdir())

        for trip_path in folders:
            if not trip_path.is_dir():
                continue

            print(f"Processing {partition}: {trip_path.name}")

            try:
                features, labels = process_trip(
                    trip_path, partition
                )
                all_features.append(features)
                all_labels.append(labels)

            except Exception as error:
                print("ERROR:", trip_path.name, error)

        if all_features:
            combined = pd.concat(
                all_features, ignore_index=True
            )

            combined.to_csv(
                OUT / f"{partition}_features.csv",
                index=False
            )

            print(
                f"\n{partition} features:",
                len(combined)
            )

        if all_labels:
            combined_labels = pd.concat(
                all_labels, ignore_index=True
            )

            combined_labels.to_csv(
                OUT / f"{partition}_labels.csv",
                index=False
            )

            print(
                f"{partition} reference labels:",
                len(combined_labels)
            )

if __name__ == "__main__":
    main()