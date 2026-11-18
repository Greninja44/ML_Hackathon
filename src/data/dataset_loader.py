"""Single entry point for loading either data workflow.

* Workflow A — Person 1's preprocessed CSVs  (``load_csv``)
* Workflow B — raw trip files                (``load_raw``, ``iter_raw``)

Both refuse the evaluator partition; only the API service reads it
(through ``src.data.io.iter_partition(..., allow_eval=True)``).
"""
from __future__ import annotations

from typing import Iterator

from src.config import load_config
from src.csv_pipeline.load import CSVPartition, load_partition as _load_csv
from src.data.io import TripData, iter_partition, list_partition, load_trip


def load_csv(partition: str, cfg: dict | None = None) -> CSVPartition:
    """Person 1's (features, labels) CSVs for ``train`` or ``student_test``, validated."""
    return _load_csv(cfg or load_config(), partition)


def iter_raw(partition: str, cfg: dict | None = None, with_route: bool = True) -> Iterator[TripData]:
    """Raw trips of a development partition, one ``TripData`` at a time."""
    return iter_partition(cfg or load_config(), partition, with_route=with_route)


def load_raw(partition: str, trip_id: str, cfg: dict | None = None, with_route: bool = True) -> TripData:
    cfg = cfg or load_config()
    if partition == "eval_test":
        raise PermissionError("Evaluator trips are read only by the API service")
    files = list_partition(cfg, partition)
    if trip_id not in files:
        raise KeyError(f"Trip {trip_id!r} not in {partition!r}")
    return load_trip(files[trip_id], trip_id, partition, with_route=with_route)


def raw_trip_ids(partition: str, cfg: dict | None = None) -> list[str]:
    return sorted(list_partition(cfg or load_config(), partition))
