"""Raw-data loading.

File formats follow Dataset Exploration §3 exactly:

* ``.track``  — whitespace separated, no header: longitude, latitude, timestamp [s]
* ``.nodes``  — whitespace separated, no header: longitude, latitude (row index = node id)
* ``.arcs``   — whitespace separated, no header: from_node_id, to_node_id (row index = arc id)
* ``.route``  — one integer per line: chronological ground-truth arc ids (optional)

This module is the interface boundary with Person 1's preprocessing work: any
replacement loader only needs to return :class:`TripData` objects.

The prediction pipeline only ever receives :class:`TripInputs` (no route), so
reference labels cannot leak into predictions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import DEVELOPMENT_PARTITIONS, partition_dir

log = logging.getLogger(__name__)

TRACK_COLUMNS = ["lon", "lat", "t"]
NODE_COLUMNS = ["lon", "lat"]
ARC_COLUMNS = ["from_node", "to_node"]
EXTENSIONS = (".track", ".nodes", ".arcs", ".route")


class DataFormatError(ValueError):
    """Raised when a raw file does not match the documented format."""


@dataclass
class TripInputs:
    """Everything the predictor is allowed to see for a trip."""

    trip_id: str
    track: pd.DataFrame  # columns lon, lat, t
    nodes: np.ndarray  # (n_nodes, 2) lon, lat
    arcs: np.ndarray  # (n_arcs, 2) int from_node, to_node


@dataclass
class TripData:
    trip_id: str
    partition: str | None
    track: pd.DataFrame
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None
    files: dict[str, Path] = field(default_factory=dict)

    def inputs(self) -> TripInputs:
        return TripInputs(self.trip_id, self.track, self.nodes, self.arcs)

    @property
    def has_route(self) -> bool:
        return self.route is not None and len(self.route) > 0


def _read_table(path: Path, n_cols: int, names: list[str], dtype) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", comment="#")
    except pd.errors.EmptyDataError:
        return pd.DataFrame({n: pd.Series(dtype=dtype) for n in names})
    if df.shape[1] != n_cols:
        raise DataFormatError(f"{path.name}: expected {n_cols} columns, found {df.shape[1]}")
    df.columns = names
    try:
        return df.astype(dtype)
    except (ValueError, TypeError) as exc:
        raise DataFormatError(f"{path.name}: non-numeric values ({exc})") from exc


def load_track(path: Path) -> pd.DataFrame:
    return _read_table(path, 3, TRACK_COLUMNS, float)


def load_nodes(path: Path) -> np.ndarray:
    return _read_table(path, 2, NODE_COLUMNS, float).to_numpy()


def load_arcs(path: Path) -> np.ndarray:
    return _read_table(path, 2, ARC_COLUMNS, np.int64).to_numpy()


def load_route(path: Path) -> np.ndarray:
    return _read_table(path, 1, ["arc_id"], np.int64)["arc_id"].to_numpy()


def _find_sibling(track_path: Path, ext: str) -> Path | None:
    same_stem = track_path.with_suffix(ext)
    if same_stem.exists():
        return same_stem
    others = sorted(track_path.parent.glob(f"*{ext}"))
    n_tracks = len(list(track_path.parent.glob("*.track")))
    if len(others) == 1 and n_tracks == 1:
        return others[0]
    return None


def trip_id_for(track_path: Path) -> str:
    """Trip id = file stem, or folder name when files have generic names."""
    n_tracks = len(list(track_path.parent.glob("*.track")))
    if n_tracks == 1 and track_path.stem.lower() in {"track", "gps", "trip"}:
        return track_path.parent.name
    return track_path.stem


def discover_trips(root: Path) -> dict[str, dict[str, Path]]:
    """Map trip_id -> {ext: path} for every ``.track`` file under ``root``."""
    root = Path(root)
    if not root.exists():
        return {}
    found: dict[str, dict[str, Path]] = {}
    for track in sorted(root.rglob("*.track")):
        tid = trip_id_for(track)
        if tid in found:
            raise DataFormatError(f"Duplicate trip id {tid!r} under {root}")
        files = {".track": track}
        for ext in EXTENSIONS[1:]:
            sib = _find_sibling(track, ext)
            if sib is not None:
                files[ext] = sib
        found[tid] = files
    return found


def load_trip(files: dict[str, Path], trip_id: str, partition: str | None = None,
              with_route: bool = True) -> TripData:
    for ext in (".track", ".nodes", ".arcs"):
        if ext not in files:
            raise DataFormatError(f"Trip {trip_id}: missing {ext} file")
    route = None
    if with_route and ".route" in files:
        route = load_route(files[".route"])
    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=load_track(files[".track"]),
        nodes=load_nodes(files[".nodes"]),
        arcs=load_arcs(files[".arcs"]),
        route=route,
        files=dict(files),
    )


def list_partition(cfg: dict, partition: str) -> dict[str, dict[str, Path]]:
    return discover_trips(partition_dir(cfg, partition))


def iter_partition(cfg: dict, partition: str, with_route: bool = True,
                   allow_eval: bool = False, trip_ids: list[str] | None = None) -> Iterator[TripData]:
    """Yield trips in a partition.

    ``eval_test`` is refused unless ``allow_eval=True``; only the API's
    server-side evaluation sets that flag.
    """
    if partition not in DEVELOPMENT_PARTITIONS and not allow_eval:
        raise PermissionError(
            f"Partition {partition!r} is reserved for final evaluation and may not be read "
            "by development code (Dataset Exploration §7)."
        )
    listing = list_partition(cfg, partition)
    for tid in trip_ids or sorted(listing):
        if tid not in listing:
            raise KeyError(f"Trip {tid!r} not found in partition {partition!r}")
        try:
            yield load_trip(listing[tid], tid, partition, with_route=with_route)
        except DataFormatError as exc:
            log.error("Skipping trip %s: %s", tid, exc)
