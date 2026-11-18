"""One entry point that turns raw trip inputs into model-ready structures.

    TripInputs ──► projection ──► RoadNetwork
              └──► CleanTrack ──► CandidateSet ──► feature DataFrame

Used identically by training, evaluation and the API. Labels are attached
separately (``attach_labels``) and only when a reference route is available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.features import build_features
from src.data.io import TripData, TripInputs
from src.data.labels import RouteAlignment, align_route
from src.data.network import RoadNetwork
from src.data.preprocess import CleanTrack, clean_track
from src.geo import LocalProjection
from src.map_matching.candidates import CandidateSet, generate_candidates

log = logging.getLogger(__name__)


@dataclass
class PreparedTrip:
    trip_id: str
    net: RoadNetwork
    track: CleanTrack
    cands: CandidateSet
    features: pd.DataFrame


def build_network(inputs: TripInputs, cfg: dict) -> RoadNetwork:
    lon = inputs.track["lon"].to_numpy(float)
    lat = inputs.track["lat"].to_numpy(float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.sum() == 0:
        lon, lat = inputs.nodes[:, 0], inputs.nodes[:, 1]
        ok = np.isfinite(lon) & np.isfinite(lat)
    proj = LocalProjection.for_lonlat(lon[ok], lat[ok])
    ncfg = cfg["network"]
    return RoadNetwork.from_arrays(inputs.nodes, inputs.arcs, proj,
                                   directed=ncfg["directed"], densify_step_m=ncfg["densify_step_m"])


def prepare_trip(inputs: TripInputs, cfg: dict, net: RoadNetwork | None = None) -> PreparedTrip:
    net = net or build_network(inputs, cfg)
    c = cfg["candidates"]
    track = clean_track(inputs.track, net.proj, heading_window=c["heading_window"])
    cands = generate_candidates(track, net, c["radius_m"], c["max_candidates"], c["fallback_radius_m"])
    feats = build_features(track, cands, net)
    feats.insert(0, "trip_id", inputs.trip_id)
    return PreparedTrip(inputs.trip_id, net, track, cands, feats)


@dataclass
class TripLabels:
    alignment: RouteAlignment
    label_arc: np.ndarray  # per cleaned observation
    is_ref: np.ndarray  # per candidate pair (0/1)


def attach_labels(prep: PreparedTrip, trip: TripData) -> TripLabels:
    if not trip.has_route:
        raise ValueError(f"Trip {trip.trip_id} has no reference route")
    al = align_route(prep.track.xy, trip.route, prep.net)
    is_ref = (prep.cands.arc == al.label_arc[prep.cands.obs]).astype(int)
    return TripLabels(al, al.label_arc, is_ref)


def prepare_labelled(trip: TripData, cfg: dict) -> tuple[PreparedTrip, TripLabels]:
    prep = prepare_trip(trip.inputs(), cfg)
    labels = attach_labels(prep, trip)
    prep.features["label"] = labels.is_ref
    prep.features["label_residual_m"] = labels.alignment.residual_m[prep.cands.obs]
    return prep, labels
