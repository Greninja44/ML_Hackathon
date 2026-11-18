"""Trip-wise evaluation: classifier-only and classifier+HMM on a set of labelled trips.

Shared by model training (student-test comparison), hyperparameter optimisation
and the decoder comparison so every number is computed the same way.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.io import TripData, iter_partition
from src.data.prepare import PreparedTrip, TripLabels, prepare_labelled
from src.map_matching.metrics import candidate_miss_rate, sequence_metrics, top1_accuracy
from src.map_matching.reconstruction import RouteReconstructor
from src.models.evaluate import pair_metrics


@dataclass
class LabelledTrip:
    trip: TripData
    prep: PreparedTrip
    labels: TripLabels


def load_labelled_trips(cfg: dict, partition: str) -> list[LabelledTrip]:
    out = []
    for trip in iter_partition(cfg, partition):
        if trip.has_route:
            prep, lab = prepare_labelled(trip, cfg)
            out.append(LabelledTrip(trip, prep, lab))
    return out


def evaluate_on_trips(trips: list[LabelledTrip], cfg: dict, scorer=None, modes=("classifier", "hmm")) -> tuple[dict, pd.DataFrame]:
    """Returns (aggregate metrics, per-trip DataFrame).

    Aggregation: Top-1 is pooled over all labelled observations; continuity
    and sequence metrics are averaged over trips; pair metrics pooled over pairs.
    """
    rec = RouteReconstructor(cfg, scorer)
    rows, y_all, p_all = [], [], []
    infer_s, n_pairs = 0.0, 0
    correct = {m: 0 for m in modes}
    n_lab = 0
    for lt in trips:
        prep, lab = lt.prep, lt.labels
        t0 = time.perf_counter()
        proba = scorer.predict_candidates(prep.features) if scorer is not None else None
        infer_s += time.perf_counter() - t0
        n_pairs += len(prep.features)
        if proba is not None:
            y_all.append(lab.is_ref)
            p_all.append(proba)
        mask = lab.label_arc >= 0
        n_lab += int(mask.sum())
        row = {"trip_id": lt.trip.trip_id, "n_obs": len(prep.track),
               "candidate_miss_rate": candidate_miss_rate(prep.cands.obs, prep.cands.arc, lab.label_arc)}
        for mode in modes:
            r = rec.reconstruct(lt.trip.inputs(), mode, prepared=prep, proba=proba)
            correct[mode] += int((r.pred_arc[mask] == lab.label_arc[mask]).sum())
            sm = sequence_metrics(r.route_arcs, lt.trip.route, prep.net)
            row.update({
                f"{mode}_top1": top1_accuracy(r.pred_arc, lab.label_arc),
                f"{mode}_continuity": r.continuity["route_continuity"],
                f"{mode}_breaks": r.n_breaks,
                f"{mode}_decode_ms": r.timings_ms["decode_ms"],
                **{f"{mode}_{k}": v for k, v in sm.items()},
            })
        rows.append(row)
    per_trip = pd.DataFrame(rows)
    agg: dict = {"n_trips": len(trips), "n_labelled_obs": n_lab}
    for mode in modes:
        agg[f"{mode}_top1"] = correct[mode] / n_lab if n_lab else float("nan")
        for k in ("continuity", "seq_f1", "seq_precision", "seq_recall", "length_precision",
                  "length_recall", "edit_similarity", "decode_ms"):
            agg[f"{mode}_{k}"] = float(per_trip[f"{mode}_{k}"].mean()) if len(per_trip) else float("nan")
    agg["candidate_miss_rate"] = float(per_trip["candidate_miss_rate"].mean()) if len(per_trip) else float("nan")
    if p_all:
        pm = pair_metrics(np.concatenate(y_all), np.concatenate(p_all))
        agg.update({f"pair_{k}": v for k, v in pm.items()})
        agg["inference_ms_per_1k_pairs"] = 1e3 * infer_s / max(n_pairs, 1) * 1000
        agg["inference_ms_per_trip"] = 1e3 * infer_s / max(len(trips), 1)
    return agg, per_trip
