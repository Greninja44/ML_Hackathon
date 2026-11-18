"""Route reconstruction: candidates → ML scores → HMM/Viterbi → route.

Public interface::

    rec = RouteReconstructor(cfg, scorer=CandidateScorer.load(...))  # scorer optional
    result = rec.reconstruct(trip_inputs, mode="hmm")                # or "classifier"

``mode="classifier"``: each observation independently takes its highest-scoring
candidate (classifier-only baseline required by MT §9).
``mode="hmm"``: Viterbi over the candidate lattice.

If no scorer is supplied (or it fails), emissions fall back to the Gaussian
distance model and classifier-only mode falls back to "nearest arc"; the result
records which method was actually used.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.data.io import TripInputs
from src.data.prepare import PreparedTrip, prepare_trip
from src.map_matching.hmm import TransitionModel, emission_scores
from src.map_matching.metrics import collapse, route_continuity
from src.map_matching.viterbi import viterbi

log = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    trip_id: str
    mode: str
    emission_source: str  # "ml", "hybrid", "gaussian"
    obs_index: np.ndarray  # raw .track row of each cleaned observation
    pred_arc: np.ndarray  # predicted arc per cleaned observation (-1 = unmatched)
    chosen: np.ndarray  # index of the chosen candidate pair per observation (-1 = unmatched)
    pred_score: np.ndarray  # classifier probability (or nan) of the chosen candidate
    route_arcs: list[int]  # connected route (gaps filled by shortest paths where possible)
    n_breaks: int
    continuity: dict
    timings_ms: dict
    prepared: PreparedTrip = field(repr=False)
    proba: np.ndarray | None = field(default=None, repr=False)


class RouteReconstructor:
    def __init__(self, cfg: dict, scorer=None):
        self.cfg = cfg
        self.scorer = scorer

    def _score(self, prep: PreparedTrip) -> np.ndarray | None:
        if self.scorer is None:
            return None
        try:
            return self.scorer.predict_candidates(prep.features)
        except Exception:  # never let a model failure take down reconstruction
            log.exception("Candidate scorer failed for trip %s; falling back to distance model", prep.trip_id)
            return None

    def reconstruct(self, inputs: TripInputs, mode: str = "hmm", prepared: PreparedTrip | None = None,
                    proba: np.ndarray | None = None) -> ReconstructionResult:
        if mode not in ("hmm", "classifier"):
            raise ValueError("mode must be 'hmm' or 'classifier'")
        t0 = time.perf_counter()
        prep = prepared or prepare_trip(inputs, self.cfg)
        t1 = time.perf_counter()
        if proba is None:
            proba = self._score(prep)
        t2 = time.perf_counter()
        c = prep.cands
        n_obs = c.n_obs
        hcfg = self.cfg["hmm"]
        source = hcfg["emission"] if proba is not None else "gaussian"

        if mode == "classifier":
            score = proba if proba is not None else -c.dist
            pred = np.full(n_obs, -1)
            chosen = np.full(n_obs, -1)
            for i in range(n_obs):
                s = c.for_obs(i)
                if s.stop > s.start:
                    k = s.start + int(np.argmax(score[s]))
                    pred[i], chosen[i] = c.arc[k], k
            n_breaks = 0
            source = "ml" if proba is not None else "nearest"
        else:
            emis = emission_scores(c, hcfg, proba)
            trans = TransitionModel(prep.net, prep.track, c, hcfg)
            vr = viterbi([emis[c.for_obs(i)] for i in range(n_obs)], trans.matrix)
            chosen = np.where(vr.state >= 0, c.offsets[:-1] + vr.state, -1)
            pred = np.where(chosen >= 0, c.arc[np.maximum(chosen, 0)], -1)
            n_breaks = vr.n_breaks
        t3 = time.perf_counter()

        pred_score = np.where(chosen >= 0, proba[np.maximum(chosen, 0)], np.nan) if proba is not None \
            else np.full(n_obs, np.nan)
        route = self.assemble_route(prep, pred)
        cont = route_continuity(pred, prep.net, self.cfg["metrics"]["continuity_max_hops"])
        t4 = time.perf_counter()
        timings = {
            "prepare_ms": (t1 - t0) * 1e3, "score_ms": (t2 - t1) * 1e3,
            "decode_ms": (t3 - t2) * 1e3, "assemble_ms": (t4 - t3) * 1e3, "total_ms": (t4 - t0) * 1e3,
        }
        return ReconstructionResult(inputs.trip_id, mode, source, prep.track.obs_index, pred, chosen, pred_score,
                                    route, n_breaks, cont, timings, prep, proba)

    def assemble_route(self, prep: PreparedTrip, pred_arc: np.ndarray) -> list[int]:
        """De-duplicate the per-observation arcs and bridge gaps with shortest paths.

        A bridge is only inserted if the shortest path is short relative to
        the straight-line gap (``max_route_factor``), so we never fabricate long
        detours; otherwise the route simply has a discontinuity.
        """
        net = prep.net
        seq = collapse(pred_arc)
        if not seq:
            return []
        cutoff = float(self.cfg["hmm"]["max_route_slack_m"]) * 5
        out = [seq[0]]
        for b in seq[1:]:
            a = out[-1]
            path = net.node_path(int(net.arcs[a, 1]), int(net.arcs[b, 0]), cutoff)
            if path is None and not net.directed:
                path = net.node_path(int(net.arcs[a, 0]), int(net.arcs[b, 1]), cutoff)
            if path:
                out.extend(p for p in path if p != out[-1])
            if out[-1] != b:
                out.append(b)
        return out
