"""Run dataset exploration on the development partitions.

    python scripts/run_eda.py [--maps 3]

Writes tables, plots and Folium maps to results/eda/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve  # noqa: E402
from src.data.exploration import explore, plot_trip  # noqa: E402
from src.data.io import iter_partition  # noqa: E402
from src.data.prepare import build_network  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.visualization.folium_maps import save_map, trip_map  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=3, help="number of training trips to map")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging("eda", cfg["paths"]["results_dir"])
    out = resolve(cfg["paths"]["results_dir"]) / "eda"
    tables = explore(cfg, out)
    if "partition_summary" in tables:
        print(tables["partition_summary"].to_string(index=False))
    else:
        print("No development trips found — place the data under", resolve(cfg["data"]["raw_dir"]))
        return
    for i, trip in enumerate(iter_partition(cfg, "train")):
        if i >= args.maps:
            break
        plot_trip(trip, out / "trips" / f"{trip.trip_id}.png")
        net = build_network(trip.inputs(), cfg)
        m = trip_map(net, trip.track[["lon", "lat"]].to_numpy(), trip.route,
                     title=f"Training trip {trip.trip_id} (raw GPS + reference route)")
        save_map(m, out / "maps" / f"{trip.trip_id}.html")
    print("EDA written to", out)


if __name__ == "__main__":
    main()
