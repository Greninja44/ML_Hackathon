"""CSV-workflow experiments on Person 1's preprocessed data.

    python -m src.csv_pipeline.experiments prepare
    python -m src.csv_pipeline.experiments compare            [--models a,b]
    python -m src.csv_pipeline.experiments optimize           [--model m --trials 30]
    python -m src.csv_pipeline.experiments decoders

Data use: 80 training trips → GroupKFold(trip) CV, Optuna, decoder tuning;
10 student-testing trips → comparison only. Evaluator data is not part of
this workflow. All outputs: results/csv_pipeline/, models/csv_experiments/.

Pre-declared choices (fixed before looking at student-test numbers):
* target = monotone route alignment label (labels.py);
* primary metric = mean CV observation-level Top-1 over the 80 training trips
  (all points, misses count as wrong); tie-breaker = CV log loss;
* sensitivity metric = the same excluding training trips whose alignment miss
  rate exceeds 50 % (data-quality anomalies; reported, never used to select);
* HMM parameters for the model comparison = TopoHMMParams defaults; the final
  model's decoder is tuned separately on out-of-fold training probabilities.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.csv_pipeline.features import CSV_FEATURES, build_features  # noqa: E402
from src.csv_pipeline.hmm import TopoHMMParams, argmax_trip, decode_trip, route_metrics  # noqa: E402
from src.csv_pipeline.labels import align_partition, pair_labels  # noqa: E402
from src.csv_pipeline.load import load_partition  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.models.evaluate import evaluate, pair_metrics, top1_from_pairs  # noqa: E402
from src.models.inference import CandidateScorer  # noqa: E402
from src.models.registry import default_specs  # noqa: E402
from src.models.train import cross_validate, fit_pipeline, predict  # noqa: E402
from src.visualization import plots  # noqa: E402

log = logging.getLogger("csv")
FLAG_MISS_RATE = 0.5


# ------------------------------------------------------------------ data
def _paths(cfg):
    res = resolve(cfg["paths"]["results_dir"]) / "csv_pipeline"
    mod = resolve(cfg["paths"]["models_dir"]) / "csv_experiments"
    cache = resolve(cfg["data"]["cache_dir"]) / "csv"
    for p in (res, mod, cache):
        p.mkdir(parents=True, exist_ok=True)
    return res, mod, cache


def prepare(cfg, use_cache=True) -> dict:
    res, _, cache = _paths(cfg)
    out = {}
    for part in ("train", "student_test"):
        f_pairs, f_obs, f_route = (cache / f"{part}_{n}.pkl" for n in ("pairs", "obs", "route"))
        if use_cache and f_pairs.exists() and f_obs.exists() and f_route.exists():
            out[part] = (pd.read_pickle(f_pairs), pd.read_pickle(f_obs), pd.read_pickle(f_route))
            continue
        t0 = time.perf_counter()
        P = load_partition(cfg, part)
        obs = align_partition(P.pairs, P.route)
        feats = build_features(P.pairs)
        feats["label"] = pair_labels(feats, obs)
        feats = feats.rename(columns={"gps_index": "obs"})
        obs = obs.rename(columns={"gps_index": "obs"})
        feats.to_pickle(f_pairs); obs.to_pickle(f_obs); P.route.to_pickle(f_route)
        (res / f"{part}_schema_checks.json").write_text(json.dumps(P.checks, indent=2, default=int))
        log.info("%s: %d pairs, %d points, miss rate %.4f, built in %.1fs", part, len(feats), len(obs),
                 1 - obs["label_in_candidates"].mean(), time.perf_counter() - t0)
        out[part] = (feats, obs, P.route)
    return out


def flagged_trips(obs: pd.DataFrame) -> list[str]:
    miss = 1 - obs.groupby("trip_id")["label_in_candidates"].mean()
    return sorted(miss[miss > FLAG_MISS_RATE].index)


# ------------------------------------------------------------------ evaluation
def eval_trips(pairs, obs, route, proba, hmm: TopoHMMParams | None, modes=("classifier", "hmm")):
    rows = []
    totals = {m: [0, 0] for m in modes}
    for tid, g in pairs.groupby("trip_id", sort=True):
        idx = g.index.to_numpy()
        gp = g.rename(columns={"obs": "gps_index"})
        o = obs[obs["trip_id"] == tid].sort_values("obs")
        lab = o["label_arc"].to_numpy()
        r = route.loc[route["trip_id"] == tid, "true_arc_id"].to_numpy()
        p = proba[idx] if proba is not None else None
        row = {"trip_id": tid, "n_points": len(o), "miss_rate": 1 - o["label_in_candidates"].mean()}
        for mode in modes:
            t0 = time.perf_counter()
            if mode == "classifier":
                chosen = argmax_trip(gp, p if p is not None else -gp["distance_m"].to_numpy())
            else:
                chosen = decode_trip(gp, p, hmm)
            dt = time.perf_counter() - t0
            m = route_metrics(gp, chosen, r, lab)
            totals[mode][0] += int(round(m["top1"] * len(o))); totals[mode][1] += len(o)
            row.update({f"{mode}_{k}": v for k, v in m.items()})
            row[f"{mode}_decode_ms"] = dt * 1e3
        rows.append(row)
    per = pd.DataFrame(rows)
    agg = {}
    for mode in modes:
        agg[f"{mode}_top1"] = totals[mode][0] / totals[mode][1]
        for k in ("continuity", "seq_f1", "seq_precision", "seq_recall", "length_precision", "length_recall",
                  "edit_similarity", "decode_ms"):
            agg[f"{mode}_{k}"] = float(per[f"{mode}_{k}"].mean())
    return agg, per


def _excl_eval(obs_train, flagged):
    def f(pva, proba, val_trips):
        keep = [t for t in val_trips if t not in flagged]
        if len(keep) == len(val_trips):
            return {"top1_excl_flagged": None}
        mask = pva["trip_id"].isin(keep).to_numpy()
        m = top1_from_pairs(pva[mask], proba[mask], obs_train[obs_train["trip_id"].isin(keep)])
        return {"top1_excl_flagged": m["top1_accuracy"]}
    return f


def run_model(spec, params, tag, data, cfg, hmm=TopoHMMParams()):
    res, mod, _ = _paths(cfg)
    (tr_pairs, tr_obs, _), (st_pairs, st_obs, st_route) = data["train"], data["student_test"]
    flagged = flagged_trips(tr_obs)
    out = res / "models" / tag
    out.mkdir(parents=True, exist_ok=True)

    log.info("[%s] CV %s", tag, params)
    folds = cross_validate(spec, params, tr_pairs, tr_obs, cfg, feature_cols=CSV_FEATURES,
                           extra_eval=_excl_eval(tr_obs, flagged))
    folds.insert(0, "model", tag)
    folds.to_csv(out / "cv_folds.csv", index=False)

    pipe, fit_s = fit_pipeline(spec, params, tr_pairs, cfg["seed"], CSV_FEATURES)
    scorer = CandidateScorer(pipe, CSV_FEATURES, {
        "model_name": tag, "base_model": spec.name, "family": spec.family, "params": params, "loss": spec.loss,
        "seed": cfg["seed"], "data": "Person 1 CSVs (train)", "target": "monotone route alignment",
        "max_train_rows": spec.max_train_rows})
    scorer.save(mod / tag)

    rng = np.random.default_rng(cfg["seed"])
    trips = np.sort(tr_pairs["trip_id"].unique())
    sub = rng.choice(trips, size=min(20, len(trips)), replace=False)
    tp = tr_pairs[tr_pairs["trip_id"].isin(sub)]
    p_tr, _ = predict(pipe, tp, CSV_FEATURES)
    train_m = evaluate(tp, p_tr, tr_obs[tr_obs["trip_id"].isin(sub)])

    p_st, inf_s = predict(pipe, st_pairs, CSV_FEATURES)
    pm = evaluate(st_pairs, p_st, st_obs)
    route_agg, per_trip = eval_trips(st_pairs, st_obs, st_route, p_st, hmm)
    per_trip.to_csv(out / "student_test_per_trip.csv", index=False)
    plots.confusion(pm["confusion_matrix"], out / "confusion_matrix.png", f"{tag}: student-test candidate pairs")
    _diagnostics(tag, pipe, out)

    fold_excl = folds["top1_excl_flagged"].dropna() if "top1_excl_flagged" in folds else pd.Series(dtype=float)
    summary = {
        "model": tag, "base_model": spec.name, "family": spec.family, "key_config": json.dumps(params, default=str),
        "cv_macro_f1_mean": folds["macro_f1"].mean(), "cv_macro_f1_std": folds["macro_f1"].std(),
        "cv_top1_mean": folds["top1_accuracy"].mean(), "cv_top1_std": folds["top1_accuracy"].std(),
        "cv_log_loss_mean": folds["log_loss"].mean(), "cv_log_loss_std": folds["log_loss"].std(),
        "cv_top1_fold_with_flagged_excl": float(fold_excl.mean()) if len(fold_excl) else np.nan,
        "train_top1": train_m["top1_accuracy"], "train_macro_f1": train_m["macro_f1"],
        "train_log_loss": train_m["log_loss"],
        "student_top1": pm["top1_accuracy"], "student_macro_f1": pm["macro_f1"],
        "student_precision": pm["precision"], "student_recall": pm["recall"], "student_f1": pm["f1"],
        "student_accuracy": pm["accuracy"], "student_log_loss": pm["log_loss"], "student_roc_auc": pm["roc_auc"],
        "student_candidate_miss_rate": pm["candidate_miss_rate"],
        "student_route_continuity": route_agg["classifier_continuity"],
        "student_seq_f1": route_agg["classifier_seq_f1"],
        "student_hmm_top1": route_agg["hmm_top1"], "student_hmm_route_continuity": route_agg["hmm_continuity"],
        "student_hmm_seq_f1": route_agg["hmm_seq_f1"],
        "train_time_s": fit_s, "inference_ms_per_1k_pairs": 1e6 * inf_s / len(st_pairs),
        "model_size_kb": sum(f.stat().st_size for f in (mod / tag).iterdir()) / 1024,
    }
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "student_pairs": pm, "train_subset": train_m,
                                                  "route": route_agg, "flagged_train_trips": flagged,
                                                  "hmm_params": asdict(hmm)}, indent=2, default=float))
    log.info("[%s] CV top1 %.4f±%.4f | student top1 %.4f → HMM %.4f | continuity %.3f → %.3f | macroF1 %.4f",
             tag, summary["cv_top1_mean"], summary["cv_top1_std"], summary["student_top1"],
             summary["student_hmm_top1"], summary["student_route_continuity"],
             summary["student_hmm_route_continuity"], summary["student_macro_f1"])
    return summary


def _diagnostics(tag, pipe, out: Path):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        plots.feature_importance(CSV_FEATURES, clf.feature_importances_, out / "feature_importance.png",
                                 f"{tag}: impurity-based importance")
    elif hasattr(clf, "coef_"):
        plots.feature_importance(CSV_FEATURES, clf.coef_.ravel(), out / "feature_importance.png",
                                 f"{tag}: standardised coefficient")
    if hasattr(clf, "loss_curve_"):
        plots.loss_curves({"training loss": clf.loss_curve_}, out / "loss_curve.png", f"{tag}: training loss per epoch",
                          "log loss")
        if getattr(clf, "validation_scores_", None) is not None:
            plots.loss_curves({"validation accuracy": clf.validation_scores_}, out / "validation_curve.png",
                              f"{tag}: held-out validation score per epoch", "accuracy")
    if len(getattr(clf, "train_score_", [])):
        curves = {"training (neg. log loss)": list(clf.train_score_)}
        if len(getattr(clf, "validation_score_", [])):
            curves["validation (neg. log loss)"] = list(clf.validation_score_)
        plots.loss_curves(curves, out / "loss_curve.png", f"{tag}: score per boosting iteration", "negative log loss")


# ------------------------------------------------------------------ commands
def cmd_compare(cfg, data, models):
    res, _, _ = _paths(cfg)
    specs = default_specs(cfg["training"]["n_jobs"])
    names = models or list(specs)
    rows = [run_model(specs[n], dict(specs[n].params), n, data, cfg) for n in names]
    df = pd.DataFrame(rows)
    comp = res / "model_comparison.csv"
    if comp.exists() and models:
        prev = pd.read_csv(comp)
        df = pd.concat([prev[~prev["model"].isin(df["model"])], df], ignore_index=True)
    df.to_csv(comp, index=False)
    folds = pd.concat([pd.read_csv(res / "models" / m / "cv_folds.csv") for m in df["model"]], ignore_index=True)
    from src.models.train import comparison_plots
    comparison_plots(df, folds, res / "plots")
    ranked = df[~df["model"].str.endswith("_tuned")].sort_values(["cv_top1_mean", "cv_log_loss_mean"],
                                                                 ascending=[False, True])
    sel = {"primary_metric": "mean CV observation-level Top-1 (GroupKFold by trip, 80 training trips)",
           "tie_breaker": "CV log loss",
           "ranking": ranked[["model", "cv_top1_mean", "cv_top1_std", "cv_log_loss_mean", "student_top1",
                              "student_hmm_top1"]].to_dict("records"),
           "selected_model": ranked.iloc[0]["base_model"]}
    (res / "model_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    log.info("Selected: %s", sel["selected_model"])


def cmd_optimize(cfg, data, model, trials):
    import optuna
    from src.models.optimize import search_space
    res, mod, _ = _paths(cfg)
    name = model or json.loads((res / "model_selection.json").read_text())["selected_model"]
    spec = default_specs(cfg["training"]["n_jobs"])[name]
    tr_pairs, tr_obs, _ = data["train"]
    out = res / "optuna" / name
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = search_space(name, trial, spec.params)
        folds = cross_validate(spec, params, tr_pairs, tr_obs, cfg, trial=trial, feature_cols=CSV_FEATURES)
        trial.set_user_attr("cv_macro_f1", float(folds["macro_f1"].mean()))
        trial.set_user_attr("cv_log_loss", float(folds["log_loss"].mean()))
        return float(folds["top1_accuracy"].mean())

    study = optuna.create_study(direction="maximize", study_name=f"csv_{name}",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=min(8, trials)),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=trials, timeout=cfg["optuna"]["timeout_s"], gc_after_trial=True)
    wall = time.perf_counter() - t0
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out / "trials.csv", index=False)
    best = search_space(name, optuna.trial.FixedTrial(study.best_params), spec.params)
    summary = {"model": name, "sampler": "TPE (seed %d)" % cfg["seed"], "pruner": "MedianPruner",
               "objective": "mean CV Top-1 (GroupKFold by trip, 80 training trips)", "n_trials": len(study.trials),
               "trial_states": trials_df["state"].value_counts().to_dict(), "best_value": study.best_value,
               "best_params": best, "wall_time_s": wall}
    (out / "study_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(study); ax.figure.set_size_inches(8, 4.5); ax.figure.tight_layout()
        ax.figure.savefig(out / "optimization_history.png", dpi=150)
        if len(study.best_params) > 1:
            ax = plot_param_importances(study); ax.figure.tight_layout()
            ax.figure.savefig(out / "param_importances.png", dpi=150)
    except Exception:
        log.exception("Optuna plots failed")

    untuned = json.loads((res / "models" / name / "metrics.json").read_text())["summary"]
    tuned = run_model(spec, best, f"{name}_tuned", data, cfg)
    cols = ["model", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std", "cv_log_loss_mean", "student_top1",
            "student_macro_f1", "student_log_loss", "student_hmm_top1", "inference_ms_per_1k_pairs", "train_time_s"]
    pd.DataFrame([untuned, tuned])[cols].to_csv(res / "tuned_vs_untuned.csv", index=False)
    chosen = f"{name}_tuned" if tuned["cv_top1_mean"] >= untuned["cv_top1_mean"] else name
    final = mod / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(mod / chosen, final)
    (final / "final_model.json").write_text(json.dumps({
        "final_model": chosen, "rule": "tuned if CV Top-1 >= untuned CV Top-1",
        "untuned_cv_top1": untuned["cv_top1_mean"], "tuned_cv_top1": tuned["cv_top1_mean"]}, indent=2))
    comp = pd.read_csv(res / "model_comparison.csv")
    comp = pd.concat([comp[comp["model"] != tuned["model"]], pd.DataFrame([tuned])], ignore_index=True)
    comp.to_csv(res / "model_comparison.csv", index=False)
    log.info("Final CSV model: %s", chosen)


def cmd_decoders(cfg, data, n_trips=30):
    res, mod, _ = _paths(cfg)
    scorer = CandidateScorer.load(mod / "final")
    (tr_pairs, tr_obs, tr_route), (st_pairs, st_obs, st_route) = data["train"], data["student_test"]
    out = res / "decoders"
    out.mkdir(parents=True, exist_ok=True)

    # out-of-fold probabilities on training trips (each trip scored by a model that never saw it)
    groups = tr_pairs["trip_id"].to_numpy()
    X = tr_pairs[CSV_FEATURES].to_numpy(float)
    y = tr_pairs["label"].to_numpy()
    oof = np.zeros(len(tr_pairs))
    max_rows = scorer.metadata.get("max_train_rows")
    for k, (tr, va) in enumerate(GroupKFold(n_splits=cfg["training"]["cv_folds"]).split(X, groups=groups)):
        idx = tr if not max_rows or len(tr) <= max_rows else \
            np.random.default_rng(cfg["seed"] + k).choice(tr, max_rows, replace=False)
        pipe = clone(scorer.pipeline).fit(X[idx], y[idx])
        oof[va] = pipe.predict_proba(X[va])[:, list(pipe.classes_).index(1)]
    rng = np.random.default_rng(cfg["seed"])
    tune_ids = sorted(rng.choice(np.unique(groups), size=min(n_trips, len(np.unique(groups))), replace=False))
    m = tr_pairs["trip_id"].isin(tune_ids).to_numpy()
    tp, to, tr_ = tr_pairs[m], tr_obs[tr_obs["trip_id"].isin(tune_ids)], tr_route[tr_route["trip_id"].isin(tune_ids)]
    grid = []
    for emission in ("ml", "hybrid"):
        for c_next in (0.0, 0.5, 1.5):
            for c_jump in (2.0, 4.0, 8.0):
                for c_uturn in (4.0, 10.0):
                    p = TopoHMMParams(emission=emission, ml_weight=1.0 if emission == "ml" else 0.7,
                                      c_next=c_next, c_jump=c_jump, c_uturn=c_uturn)
                    agg, _ = eval_trips(tp.reset_index(drop=True), to, tr_, oof[m], p, modes=("hmm",))
                    grid.append({**asdict(p), "hmm_top1": agg["hmm_top1"], "hmm_continuity": agg["hmm_continuity"]})
                    log.info("grid %s → top1 %.4f", grid[-1], agg["hmm_top1"])
    g = pd.DataFrame(grid).sort_values("hmm_top1", ascending=False)
    g.to_csv(out / "hmm_tuning_grid.csv", index=False)
    best = TopoHMMParams(**{k: g.iloc[0][k] for k in asdict(TopoHMMParams())})
    (mod / "final" / "topo_hmm_config.json").write_text(json.dumps(asdict(best), indent=2, default=float))
    (out / "hmm_tuning_best.json").write_text(json.dumps({"tuned_on": "out-of-fold probabilities", "trip_ids": tune_ids,
                                                          "best": asdict(best)}, indent=2, default=float))

    p_st = scorer.predict_candidates(st_pairs)
    rows, per_all = [], []
    for method, proba, mode, hp in (("nearest_arc", None, "classifier", best),
                                    ("hmm_distance_only", None, "hmm", TopoHMMParams(**{**asdict(best), "emission": "gaussian"})),
                                    ("classifier_only", p_st, "classifier", best),
                                    ("classifier_hmm", p_st, "hmm", best)):
        agg, per = eval_trips(st_pairs, st_obs, st_route, proba, hp, modes=(mode,))
        rows.append({"method": method, **{k.replace(f"{mode}_", ""): v for k, v in agg.items()}})
        per.insert(0, "method", method)
        per_all.append(per.rename(columns=lambda c: c.replace(f"{mode}_", "")))
        log.info("%-18s %s", method, rows[-1])
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "decoder_comparison.csv", index=False)
    per = pd.concat(per_all, ignore_index=True)
    per.to_csv(out / "decoder_comparison_per_trip.csv", index=False)
    wide = per.pivot(index="trip_id", columns="method", values="top1")
    wide["hmm_gain"] = wide["classifier_hmm"] - wide["classifier_only"]
    wide.sort_values("hmm_gain").to_csv(out / "hmm_gain_per_trip.csv")
    plots.grouped_bars(comp.rename(columns={"method": "model"}), ["top1", "continuity", "seq_f1"],
                       ["Top-1", "Route continuity", "Sequence F1"], out / "decoder_comparison.png",
                       "Student-testing: decoding methods (CSV workflow, topology-only HMM)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "compare", "optimize", "decoders"])
    ap.add_argument("--models")
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    setup_logging(f"csv_{a.command}", cfg["paths"]["results_dir"])
    data = prepare(cfg, use_cache=not a.no_cache)
    if a.command == "compare":
        cmd_compare(cfg, data, a.models.split(",") if a.models else None)
    elif a.command == "optimize":
        cmd_optimize(cfg, data, a.model, a.trials)
    elif a.command == "decoders":
        cmd_decoders(cfg, data)


if __name__ == "__main__":
    main()
