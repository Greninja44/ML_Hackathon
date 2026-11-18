"""Geometry helpers.

Longitude/latitude are angular, so Euclidean distance in degrees is not a
physical distance (Dataset Exploration §5). Each trip is projected to the UTM
zone containing its centroid; within a single city-scale trip the UTM scale
error is < 0.1 %, so planar distances in metres are accurate enough for map
matching and much cheaper than geodesic calls in inner loops.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer


@dataclass(frozen=True)
class LocalProjection:
    epsg: int
    _fwd: Transformer
    _inv: Transformer

    @classmethod
    def for_lonlat(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon.size == 0:
            raise ValueError("Cannot choose a projection for an empty coordinate set")
        clon, clat = float(np.nanmedian(lon)), float(np.nanmedian(lat))
        zone = int((clon + 180.0) // 6.0) % 60 + 1
        epsg = (32600 if clat >= 0 else 32700) + zone
        fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        return cls(epsg, fwd, inv)

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (used for EDA summaries only)."""
    r = 6_371_008.8
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass bearing (0 = north, clockwise) of planar displacement vectors."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def project_points_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project points ``p`` onto segments ``a``→``b`` (all arrays shape (n, 2)).

    Returns (distance, t) where ``t`` in [0, 1] is the position along the
    segment of the closest point.
    """
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dist = np.linalg.norm(p - closest, axis=1)
    return dist, t
