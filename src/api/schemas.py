"""Pydantic request/response schemas.

No schema contains reference-route data: ground truth stays server-side and
only derived metrics leave the API (MT §6, AP §7).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

LatLng = tuple[float, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: Optional[str]
    emission: str
    served_partition: str
    n_trips: int
    messages: list[str] = []


class TripSummary(BaseModel):
    trip_id: str
    n_observations: int
    duration_s: float
    bbox: list[float] = Field(description="[min_lat, min_lon, max_lat, max_lon]")


class TripListResponse(BaseModel):
    partition: str
    trips: list[TripSummary]


class NetworkResponse(BaseModel):
    trip_id: str
    n_arcs: int
    arcs: list[list[LatLng]] = Field(description="road segments near the trajectory as [lat, lon] pairs")


class ReconstructRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=200)
    mode: Literal["hmm", "classifier"] = "hmm"
    compare: bool = Field(False, description="also run the other mode and return its metrics")


class Observation(BaseModel):
    index: int = Field(description="row number in the raw .track file")
    lat: float
    lon: float
    t: float
    arc_id: Optional[int] = Field(description="predicted road segment (arc id), null if unmatched")
    snapped: Optional[LatLng] = Field(description="GPS point projected onto the predicted arc")
    score: Optional[float] = Field(description="classifier probability of the chosen candidate")


class RouteMetrics(BaseModel):
    top1_accuracy: Optional[float]
    candidate_miss_rate: Optional[float]
    route_continuity: Optional[float]
    seq_precision: Optional[float]
    seq_recall: Optional[float]
    seq_f1: Optional[float]
    length_precision: Optional[float]
    length_recall: Optional[float]
    edit_similarity: Optional[float]
    n_labelled_observations: int


class ReconstructResponse(BaseModel):
    trip_id: str
    mode: str
    model_name: Optional[str]
    emission_source: str
    n_observations_raw: int
    n_observations_used: int
    n_matched: int
    n_segment_breaks: int
    route_arc_ids: list[int]
    route_polylines: list[list[LatLng]]
    observations: list[Observation]
    metrics: Optional[RouteMetrics] = Field(description="computed server-side; null if no reference data")
    comparison: Optional[dict[str, Optional[RouteMetrics]]] = None
    timings_ms: dict[str, float]
    cleaning_report: dict
