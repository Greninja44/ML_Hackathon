"""HMM formulation for map matching.

Hidden state at step t: one candidate arc of observation t, together with the
point's projection onto that arc (``t_along``). Observations: cleaned GPS fixes.

Emission scores ``log P(z_t | s)``
----------------------------------
* ``gaussian`` (baseline, no ML): zero-mean Gaussian GPS error on the
  perpendicular distance — ``-0.5 (d/σ)²``.
* ``ml``: ``log p`` where ``p`` is the classifier's probability that this
  candidate is the reference arc. A discriminative posterior ``P(s | z)``
  divided by a uniform prior over candidates is proportional to the likelihood
  ``P(z | s)`` (the "scaled likelihood" trick of hybrid HMM/ANN systems), so
  ``log p`` is a valid emission score up to a per-step constant, which does not
  change the Viterbi argmax.
* ``hybrid``: ``w·ml + (1-w)·gaussian``; ``w`` is tuned on train/student-test.

The classifier's probabilities are used **only** as emissions — never as
transition probabilities.

Transition scores ``log P(s_t | s_{t-1})``
------------------------------------------
For candidates a (step t-1) and b (step t) we compute the network driving
distance ``d_route`` from a's projected point to b's projected point and the
straight-line distance ``d_gc`` between the two GPS fixes. Following the
widely used Newson & Krumm (2009) formulation, a plausible move has
``d_route ≈ d_gc``:

    log A(a, b) = -|d_route − d_gc| / β_eff,    β_eff = β · sqrt(max(1, Δt))

β grows with the time gap because longer gaps allow larger detours.
If b is unreachable from a within ``max_route_factor·d_gc + max_route_slack_m``
the transition is impossible (``-inf``); if every transition is impossible
the Viterbi decoder starts a new segment (see viterbi.py).

Within the same arc, moving backwards by more than ``backward_tolerance_m`` is
not allowed on a directed network (it would require a U-turn).
"""
from __future__ import annotations

import numpy as np

from src.data.network import RoadNetwork
from src.data.preprocess import CleanTrack
from src.map_matching.candidates import CandidateSet

MIN_DIJKSTRA_CUTOFF_M = 300.0


def emission_scores(cands: CandidateSet, cfg_hmm: dict, proba: np.ndarray | None,
                    mode: str | None = None) -> np.ndarray:
    """Per-pair log emission score."""
    mode = mode or cfg_hmm["emission"]
    sigma = float(cfg_hmm["sigma_m"])
    floor = float(cfg_hmm["min_log_prob"])
    gauss = -0.5 * (cands.dist / sigma) ** 2
    if mode == "gaussian" or proba is None:
        return np.maximum(gauss, floor)
    logp = np.log(np.clip(proba, np.exp(floor), 1.0))
    if mode == "ml":
        return np.maximum(logp, floor)
    if mode == "hybrid":
        w = float(cfg_hmm["ml_weight"])
        return np.maximum(w * logp + (1.0 - w) * gauss, floor)
    raise ValueError(f"Unknown emission mode {mode!r}")


class TransitionModel:
    def __init__(self, net: RoadNetwork, track: CleanTrack, cands: CandidateSet, cfg_hmm: dict):
        self.net, self.track, self.cands = net, track, cands
        self.beta = float(cfg_hmm["beta_m"])
        self.factor = float(cfg_hmm["max_route_factor"])
        self.slack = float(cfg_hmm["max_route_slack_m"])
        self.back_tol = float(cfg_hmm["backward_tolerance_m"])

    def route_distance(self, a: int, pos_a: float, b: int, pos_b: float, cutoff: float) -> float:
        """Driving distance from offset ``pos_a`` on arc a to offset ``pos_b`` on arc b (metres)."""
        net = self.net
        best = np.inf
        if a == b:
            fwd = pos_b - pos_a
            if fwd >= -self.back_tol:
                return max(fwd, 0.0)
            if not net.directed:
                return abs(fwd)
        la, lb = net.arc_len[a], net.arc_len[b]
        fa, ta = int(net.arcs[a, 0]), int(net.arcs[a, 1])
        fb, tb = int(net.arcs[b, 0]), int(net.arcs[b, 1])
        exits = [(ta, la - pos_a)]
        entries = [(fb, pos_b)]
        if not net.directed:
            exits.append((fa, pos_a))
            entries.append((tb, lb - pos_b))
        for node_out, cost_out in exits:
            if cost_out > cutoff:
                continue
            for node_in, cost_in in entries:
                d = cost_out + net.node_distance(node_out, node_in, max(cutoff, MIN_DIJKSTRA_CUTOFF_M)) + cost_in
                best = min(best, d)
        return best if best <= cutoff else np.inf

    def matrix(self, t: int) -> np.ndarray:
        c, tr, net = self.cands, self.track, self.net
        sp, sc = c.for_obs(t - 1), c.for_obs(t)
        arcs_p, arcs_c = c.arc[sp], c.arc[sc]
        pos_p = c.t_along[sp] * net.arc_len[arcs_p]
        pos_c = c.t_along[sc] * net.arc_len[arcs_c]
        d_gc = float(np.hypot(*(tr.xy[t] - tr.xy[t - 1])))
        dt = tr.dt[t] if np.isfinite(tr.dt[t]) else 1.0
        beta = self.beta * np.sqrt(max(1.0, dt))
        cutoff = self.factor * d_gc + self.slack
        A = np.full((len(arcs_p), len(arcs_c)), -np.inf)
        for i, (a, pa) in enumerate(zip(arcs_p, pos_p)):
            for j, (b, pb) in enumerate(zip(arcs_c, pos_c)):
                d = self.route_distance(int(a), float(pa), int(b), float(pb), cutoff)
                if np.isfinite(d):
                    A[i, j] = -abs(d - d_gc) / beta
        return A
