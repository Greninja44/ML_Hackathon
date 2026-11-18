"""Candidate road-segment generation.

For each cleaned GPS observation, every arc whose straight-line segment lies
within ``radius_m`` (planar metres) is a candidate, keeping the
``max_candidates`` closest. If none is found, the radius is widened once to
``fallback_radius_m`` (handles GPS outliers / sparse networks). Observations
that still have no candidates are reported and handled by the decoder.

The candidate list *is* the HMM hidden-state set for that observation, and each
candidate carries its arc id, so the candidate→road-segment correspondence is
preserved end-to-end (MT §9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.network import RoadNetwork
from src.data.preprocess import CleanTrack


@dataclass
class CandidateSet:
    obs: np.ndarray  # (P,) position of the observation in the CleanTrack
    arc: np.ndarray  # (P,) arc id
    dist: np.ndarray  # (P,) perpendicular distance in metres
    t_along: np.ndarray  # (P,) projection position along arc in [0, 1]
    fallback: np.ndarray  # (P,) bool, found only with the widened radius
    offsets: np.ndarray  # (n_obs + 1,) CSR-style offsets into the pair arrays

    @property
    def n_obs(self) -> int:
        return len(self.offsets) - 1

    def for_obs(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i + 1])

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)


def generate_candidates(track: CleanTrack, net: RoadNetwork, radius_m: float = 50.0,
                        max_candidates: int = 8, fallback_radius_m: float | None = 150.0) -> CandidateSet:
    obs, arc, dist, tal, fb = [], [], [], [], []
    offsets = [0]
    for i, p in enumerate(track.xy):
        ids, d, t = net.arcs_near(p, radius_m)
        used_fb = False
        if len(ids) == 0 and fallback_radius_m and fallback_radius_m > radius_m:
            ids, d, t = net.arcs_near(p, fallback_radius_m)
            used_fb = len(ids) > 0
        ids, d, t = ids[:max_candidates], d[:max_candidates], t[:max_candidates]
        obs.append(np.full(len(ids), i))
        arc.append(ids)
        dist.append(d)
        tal.append(t)
        fb.append(np.full(len(ids), used_fb))
        offsets.append(offsets[-1] + len(ids))

    def cat(x, dtype):
        return np.concatenate(x).astype(dtype) if x else np.zeros(0, dtype)

    return CandidateSet(cat(obs, np.int64), cat(arc, np.int64), cat(dist, float), cat(tal, float),
                        cat(fb, bool), np.asarray(offsets, dtype=np.int64))
