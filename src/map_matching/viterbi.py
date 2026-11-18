"""Log-space Viterbi decoding with segment breaks.

Recurrence (all quantities are log-probabilities, so products become sums and
there is no numerical underflow on long trips):

    δ_0(j)   = e_0(j)
    δ_t(j)   = e_t(j) + max_i [ δ_{t-1}(i) + A_t(i, j) ]
    ψ_t(j)   = argmax_i [ δ_{t-1}(i) + A_t(i, j) ]

where ``e_t`` are emission scores of the candidates of observation ``t`` and
``A_t`` the transition scores between the candidates of ``t-1`` and ``t``.

Breaks: if observation ``t`` has no candidates, or every transition into it is
impossible (``-inf``, e.g. disconnected road segments), the current segment is
closed (back-tracked from its best final state) and decoding restarts at ``t``
from its emissions alone. This always returns a decision for every observation
that has at least one candidate, instead of failing on the whole trip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG_INF = -np.inf


@dataclass
class ViterbiResult:
    state: np.ndarray  # (T,) chosen candidate index per step, -1 if none
    segment: np.ndarray  # (T,) segment id per step, -1 if no candidates
    n_breaks: int
    log_score: float  # sum of segment scores


def viterbi(emissions: list[np.ndarray], transition_fn) -> ViterbiResult:
    """Decode.

    ``emissions[t]`` — 1-D array of log emission scores (may be empty).
    ``transition_fn(t)`` — returns a (K_{t-1}, K_t) log-transition matrix for t ≥ 1.
    It is only called when both neighbouring steps have candidates.
    """
    T = len(emissions)
    state = np.full(T, -1, dtype=np.int64)
    segment = np.full(T, -1, dtype=np.int64)
    if T == 0:
        return ViterbiResult(state, segment, 0, 0.0)

    seg_id = -1
    n_breaks = 0
    total = 0.0
    delta = None
    psi: list[np.ndarray | None] = [None] * T
    seg_start = 0

    def close(end: int):
        nonlocal total
        if delta is None:
            return
        j = int(np.argmax(delta))
        total += float(delta[j])
        for t in range(end, seg_start - 1, -1):
            state[t] = j
            segment[t] = seg_id
            if t > seg_start:
                j = int(psi[t][j])

    for t in range(T):
        e = np.asarray(emissions[t], dtype=float)
        if e.size == 0:
            if delta is not None:
                close(t - 1)
                n_breaks += 1
            delta = None
            continue
        if delta is None:
            seg_id += 1
            seg_start = t
            delta = e.copy()
            continue
        A = transition_fn(t)
        scores = delta[:, None] + A  # (K_prev, K_cur)
        best_prev = np.argmax(scores, axis=0)
        best = scores[best_prev, np.arange(scores.shape[1])]
        if not np.isfinite(best).any():
            close(t - 1)
            n_breaks += 1
            seg_id += 1
            seg_start = t
            delta = e.copy()
            continue
        psi[t] = best_prev
        delta = best + e
    if delta is not None:
        close(T - 1)
    return ViterbiResult(state, segment, n_breaks, total)
