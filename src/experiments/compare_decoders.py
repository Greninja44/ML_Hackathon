"""Decoder experiments: classifier-only vs classifier+HMM (MT §9, Deliverables §5).

    python -m src.experiments.compare_decoders            # evaluate on student-testing trips
    python -m src.experiments.compare_decoders --tune     # first grid-search HMM parameters

Tuning (``--tune``) uses **out-of-fold** classifier probabilities on a seeded
subset of *training* trips: each trip is scored by a model that never saw it,
so the probabilities have realistic (not over-confident) calibration. The best
configuration is saved to ``models/final/hmm_config.json`` and used by the API.
Student-testing trips are then used for the reported comparison. The evaluator
partition is never read.

Methods compared on student-testing trips:
  nearest_arc        – no ML, no HMM (closest candidate)
  hmm_gaussian       – HMM with distance-only emissions (no ML)
  classifier_only    – final model, argmax per observation
  classifier_hmm     – final model probabilities as HMM emissions
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import load_config, resolve
from src.data.features import FEATURE_COLUMNS
from src.experiments.trip_eval import evaluate_on_trips, load_labelled_trips
from src.logging_utils import setup_logging
from src.map_matching.reconstruction import RouteReconstructor
from src.models.dataset import build_partition
from src.models.inference import CandidateScorer
from src.visualization import plots
from src.visualization.folium_maps import save_map, trip_map

log = logging.getLogger("decoders")

GRID = {
    "emission": ["ml", "hybrid"],
    "ml_weight": [0.5, 0.75],
    "sigma_m": [5.0, 8.0, 12.0],
    "beta_m": [2.0, 5.0, 10.0, 20.0, 40.0],
}


class _FixedProba:
    """Wraps precomputed out-of-fold probabilities in the scorer interface."""

    def __init__(self, table: dict):
        self.table = table

    def predict_candidates(self, features: pd.DataFrame) -> np.ndarray:
        trip = features["trip_id"].iloc[0]
        return self.table[trip][(features["obs"].to_numpy(), features["arc"].to_numpy())]


def _oof_probabilities(cfg: dict, scorer: CandidateScorer, trip_ids: list[str]) -> dict:
    train = build_partition(cfg, "train")
    pairs = train.pairs
    groups = pairs["trip_id"].to_numpy()
    oof = np.full(len(pairs), np.nan)
    from sklearn.base import clone
    n_splits = min(cfg["training"]["cv_folds"], len(np.unique(groups)))
    for tr, va in GroupKFold(n_splits=n_splits).split(pairs, groups=groups):
        if not np.isin(groups[va], trip_ids).any():
            continue
        pipe = clone(scorer.pipeline)
        X = pairs[FEATURE_COLUMNS].to_numpy(float)
        max_rows = scorer.metadata.get("max_train_rows")
        idx = tr if not max_rows or len(tr) <= max_rows else \
            np.random.default_rng(cfg["seed"]).choice(tr, max_rows, replace=False)
        pipe.fit(X[idx], pairs["label"].to_numpy()[idx])
        oof[va] = pipe.predict_proba(X[va])[:, list(pipe.classes_).index(1)]
    table = {}
    for tid in trip_ids:
        sub = pairs[pairs["trip_id"] == tid]
        p = oof[sub.index.to_numpy()]
        table[tid] = _Lookup(sub["obs"].to_numpy(), sub["arc"].to_numpy(), p)
    return table


class _Lookup:
    def __init__(self, obs, arc, p):
        self.map = {(int(o), int(a)): float(v) for o, a, v in zip(obs, arc, p)}

    def __getitem__(self, key):
        obs, arc = key
        return np.array([self.map.get((int(o), int(a)), np.nan) for o, a in zip(obs, arc)])


def tune(cfg: dict, scorer: CandidateScorer, n_trips: int, out: Path) -> dict:
    train_ids = sorted(build_partition(cfg, "train").pairs["trip_id"].unique())
    rng = np.random.default_rng(cfg["seed"])
    chosen = sorted(rng.choice(train_ids, size=min(n_trips, len(train_ids)), replace=False).tolist())
    log.info("Tuning HMM on %d training trips with out-of-fold probabilities", len(chosen))
    oof = _FixedProba(_oof_probabilities(cfg, scorer, chosen))
    trips = [t for t in load_labelled_trips(cfg, "train") if t.trip.trip_id in chosen]
    rows = []
    seen = set()
    for emission, w, sigma, beta in itertools.product(*GRID.values()):
        if emission == "ml":
            w = 1.0
        key = (emission, w, sigma, beta)
        if key in seen:
            continue
        seen.add(key)
        hcfg = dict(cfg["hmm"], emission=emission, ml_weight=w, sigma_m=sigma, beta_m=beta)
        c2 = dict(cfg, hmm=hcfg)
        for t in trips:
            t.prep.net.clear_cache()
        agg, _ = evaluate_on_trips(trips, c2, oof, modes=("hmm",))
        rows.append({"emission": emission, "ml_weight": w, "sigma_m": sigma, "beta_m": beta,
                     "hmm_top1": agg["hmm_top1"], "hmm_continuity": agg["hmm_continuity"],
                     "hmm_seq_f1": agg["hmm_seq_f1"]})
        log.info("  %s", rows[-1])
    df = pd.DataFrame(rows).sort_values("hmm_top1", ascending=False)
    df.to_csv(out / "hmm_tuning_grid.csv", index=False)
    best = df.iloc[0]
    hcfg = dict(cfg["hmm"], emission=best["emission"], ml_weight=float(best["ml_weight"]),
                sigma_m=float(best["sigma_m"]), beta_m=float(best["beta_m"]))
    meta = {"tuned_on": "out-of-fold probabilities, training trips", "trip_ids": chosen,
            "objective": "pooled observation Top-1 after HMM decoding", "hmm": hcfg}
    (out / "hmm_tuning_best.json").write_text(json.dumps(meta, indent=2))
    return hcfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-trips", type=int, default=20)
    ap.add_argument("--maps", type=int, default=10, help="number of student-test trips to map")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("compare_decoders", cfg["paths"]["results_dir"])
    final_dir = resolve(cfg["paths"]["final_model_dir"])
    scorer = CandidateScorer.load(final_dir)
    out = resolve(cfg["paths"]["results_dir"]) / "decoders"
    out.mkdir(parents=True, exist_ok=True)

    hmm_path = final_dir / "hmm_config.json"
    if args.tune:
        cfg["hmm"] = tune(cfg, scorer, args.tune_trips, out)
        hmm_path.write_text(json.dumps(cfg["hmm"], indent=2))
    elif hmm_path.exists():
        cfg["hmm"] = json.loads(hmm_path.read_text())
    log.info("HMM config: %s", cfg["hmm"])

    student = load_labelled_trips(cfg, "student_test")
    rows, per_trip_all = [], []
    for method, sc, mode in (("nearest_arc", None, "classifier"), ("hmm_gaussian", None, "hmm"),
                             ("classifier_only", scorer, "classifier"), ("classifier_hmm", scorer, "hmm")):
        agg, per_trip = evaluate_on_trips(student, cfg, sc, modes=(mode,))
        row = {"method": method}
        row.update({k.replace(f"{mode}_", ""): v for k, v in agg.items()
                    if k.startswith(mode) or k in ("candidate_miss_rate", "n_labelled_obs")})
        rows.append(row)
        per_trip.insert(0, "method", method)
        per_trip_all.append(per_trip.rename(columns=lambda c: c.replace(f"{mode}_", "")))
        log.info("%-16s top1=%.4f continuity=%.4f seqF1=%.4f", method, row["top1"], row["continuity"],
                 row["seq_f1"])
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "decoder_comparison.csv", index=False)
    pt = pd.concat(per_trip_all, ignore_index=True)
    pt.to_csv(out / "decoder_comparison_per_trip.csv", index=False)
    comp_plot = comp.rename(columns={"method": "model"})
    plots.grouped_bars(comp_plot, ["top1", "continuity", "seq_f1"], ["Top-1", "Route continuity", "Sequence F1"],
                       out / "decoder_comparison.png", "Student-testing: decoding methods")

    # per-trip HMM gain (where did HMM help / hurt?)
    wide = pt.pivot(index="trip_id", columns="method", values="top1")
    wide["hmm_gain"] = wide["classifier_hmm"] - wide["classifier_only"]
    wide.sort_values("hmm_gain").to_csv(out / "hmm_gain_per_trip.csv")

    # Folium maps for the report (reference route is shown here, offline only)
    rec = RouteReconstructor(cfg, scorer)
    for lt in student[: args.maps]:
        preds = {}
        for mode, label in (("classifier", "classifier only"), ("hmm", "classifier + HMM")):
            r = rec.reconstruct(lt.trip.inputs(), mode, prepared=lt.prep)
            preds[label] = r.route_arcs
        m = trip_map(lt.prep.net, lt.trip.track[["lon", "lat"]].to_numpy(), lt.trip.route, preds,
                     title=f"Student-test trip {lt.trip.trip_id}")
        save_map(m, out / "maps" / f"{lt.trip.trip_id}.html")
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
