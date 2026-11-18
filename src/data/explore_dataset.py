
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/raw")


def inspect_trip(trip_path, partition):
    trip_id = trip_path.name

    track_file = trip_path / f"{trip_id}.track"
    nodes_file = trip_path / f"{trip_id}.nodes"
    arcs_file = trip_path / f"{trip_id}.arcs"
    route_file = trip_path / f"{trip_id}.route"

    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        route_file,
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observations": len(track),
        "road_nodes": len(nodes),
        "road_segments": len(arcs),
        "route_length": len(route),
        "missing_values": track.isnull().sum().sum(),
        "duplicate_observations": track.duplicated().sum(),
        "timestamp_min": track["timestamp"].min(),
        "timestamp_max": track["timestamp"].max()
    }


def main():
    results = []

    for partition in ["train", "student_test"]:
        partition_path = DATASET_PATH / partition

        for trip_path in sorted(partition_path.iterdir()):
            if trip_path.is_dir():
                print("Inspecting:", trip_path.name)

                results.append(
                    inspect_trip(trip_path, partition)
                )

    df = pd.DataFrame(results)

    Path("outputs").mkdir(exist_ok=True)

    df.to_csv(
        "outputs/dataset_inventory.csv",
        index=False
    )

    print("\nDATASET INVENTORY")
    print(df.to_string(index=False))

    print("\nPARTITION SUMMARY")

    print(
        df.groupby("partition")[
            ["gps_observations", "road_nodes",
             "road_segments", "route_length"]
        ].sum()
    )

if __name__ == "__main__":
    main()