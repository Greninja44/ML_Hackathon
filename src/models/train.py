"""Train, cross-validate and compare all candidate-scoring models.

    python -m src.models.train                       # all models
    python -m src.models.train --models random_forest,mlp
    python -m src.models.train --no-hmm              # skip HMM evaluation per model

Data use (Models & Training Guidelines §4):
* 80 training trips → GroupKFold CV (groups = trip_id) and final fit;
* 10 student-testing trips → development-stage comparison;
* evaluator-testing trips are never read.

Outputs:
* models/experiments/<model>/            fitted pipeline + metadata
* results/models/<model>/                cv_folds.csv, metrics.json, plots
* results/model_comparison.csv           consolidated table (Deliverables §3.1)
* results/model_selection.json           primary metric, ranking, selected model
* results/plots/                         comparison plots (Deliverables §3.3)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import load_config, resolve
from src.data.features import FEATURE_COLUMNS
from src.experiments.trip_eval import evaluate_on_trips, load_labelled_trips
from src.logging_utils import setup_logging
from src.models.dataset import build_partition
from src.models.evaluate import evaluate
from src.models.inference import CandidateScorer
from src.models.registry import ModelSpec, default_specs
from src.visualization import plots

log = logging.getLogger("train")


def _subsample(X, y, max_rows, seed):
    if max_rows is None or len(X) <= max_rows:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
    return X[idx], y[idx]


def fit_pipeline(spec: ModelSpec, params: dict, pairs: pd.DataFrame, seed: int, feature_cols=None):
    X = pairs[feature_cols or FEATURE_COLUMNS].to_numpy(float)
    y = pairs["label"].to_numpy(int)
    X, y = _subsample(X, y, spec.max_train_rows, seed)
    pipe = spec.build(params, seed)
    t0 = time.perf_counter()
    pipe.fit(X, y)
    return pipe, time.perf_counter() - t0


def predict(pipe, pairs: pd.DataFrame, feature_cols=None) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = pipe.predict_proba(pairs[feature_cols or FEATURE_COLUMNS].to_numpy(float))[:, list(pipe.classes_).index(1)]
    return p, time.perf_counter() - t0


def cross_validate(spec: ModelSpec, params: dict, pairs: pd.DataFrame, obs: pd.DataFrame, cfg: dict,
                   trial=None, feature_cols=None, extra_eval=None) -> pd.DataFrame:
    """Group-aware CV: all observations of a trip stay in the same fold.

    If an Optuna ``trial`` is given, the running mean Top-1 is reported after
    each fold so unpromising trials can be pruned.
    """
    seed = cfg["seed"]
    groups = pairs["trip_id"].to_numpy()
    n_splits = min(cfg["training"]["cv_folds"], len(np.unique(groups)))
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(pairs, groups=groups)):
        ptr, pva = pairs.iloc[tr], pairs.iloc[va]
        pipe, fit_s = fit_pipeline(spec, params, ptr, seed + fold, feature_cols)
        proba, inf_s = predict(pipe, pva, feature_cols)
        val_trips = pva["trip_id"].unique()
        m = evaluate(pva, proba, obs[obs["trip_id"].isin(val_trips)])
        m.pop("confusion_matrix")
        if extra_eval is not None:  # e.g. sensitivity metrics on a subset of trips
            m.update(extra_eval(pva, proba, val_trips))
        rows.append({"fold": fold, "n_val_trips": len(val_trips), "fit_s": fit_s, "predict_s": inf_s, **m})
        if trial is not None:
            import optuna
            trial.report(float(np.mean([r["top1_accuracy"] for r in rows])), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return pd.DataFrame(rows)


def _dir_size_kb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024


def _history_plots(name: str, pipe, out: Path):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "loss_curve_"):  # MLP: per-epoch training loss (+ validation score if early stopping)
        plots.loss_curves({"training loss": clf.loss_curve_}, out / "loss_curve.png",
                          f"{name}: training loss per epoch", "log loss")
        if getattr(clf, "validation_scores_", None) is not None:
            plots.loss_curves({"validation accuracy": clf.validation_scores_}, out / "validation_curve.png",
                              f"{name}: held-out validation score per epoch", "accuracy")
    if hasattr(clf, "train_score_") and len(getattr(clf, "train_score_", [])):  # HistGradientBoosting
        curves = {"training (neg. log loss)": list(clf.train_score_)}
        if len(getattr(clf, "validation_score_", [])):
            curves["validation (neg. log loss)"] = list(clf.validation_score_)
        plots.loss_curves(curves, out / "loss_curve.png", f"{name}: score per boosting iteration",
                          "negative log loss")


def _importance(name: str, pipe, out: Path):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals, label = clf.feature_importances_, "impurity-based importance"
    elif hasattr(clf, "coef_"):
        vals, label = clf.coef_.ravel(), "standardised coefficient"
    else:
        return None
    plots.feature_importance(FEATURE_COLUMNS, vals, out / "feature_importance.png", f"{name}: {label}")
    return dict(zip(FEATURE_COLUMNS, map(float, vals)))


def run_model(spec: ModelSpec, params: dict, train, student_trips, cfg: dict, tag: str,
              do_hmm: bool = True) -> dict:
    """CV + full fit + student-test evaluation for one configuration. Returns a summary row."""
    seed = cfg["seed"]
    res_dir = resolve(cfg["paths"]["results_dir"]) / "models" / tag
    res_dir.mkdir(parents=True, exist_ok=True)
    log.info("[%s] cross-validating (%s)", tag, params)
    folds = cross_validate(spec, params, train.pairs, train.obs, cfg)
    folds.insert(0, "model", tag)
    folds.to_csv(res_dir / "cv_folds.csv", index=False)

    log.info("[%s] fitting on all %d training trips", tag, train.pairs["trip_id"].nunique())
    pipe, fit_s = fit_pipeline(spec, params, train.pairs, seed)
    model_dir = resolve(cfg["paths"]["models_dir"]) / "experiments" / tag
    scorer = CandidateScorer(pipe, FEATURE_COLUMNS, {
        "model_name": tag, "base_model": spec.name, "family": spec.family, "params": params,
        "loss": spec.loss, "seed": seed, "train_trips": int(train.pairs["trip_id"].nunique()),
        "train_rows": int(len(train.pairs)), "max_train_rows": spec.max_train_rows,
        "candidates_config": cfg["candidates"], "network_config": cfg["network"]})
    scorer.save(model_dir)

    # training-set metrics on a fixed subset of trips (overfitting evidence)
    rng = np.random.default_rng(seed)
    trips = np.sort(train.pairs["trip_id"].unique())
    sub = rng.choice(trips, size=min(20, len(trips)), replace=False)
    tp = train.pairs[train.pairs["trip_id"].isin(sub)]
    p_tr, _ = predict(pipe, tp)
    train_m = evaluate(tp, p_tr, train.obs[train.obs["trip_id"].isin(sub)])

    log.info("[%s] evaluating on %d student-testing trips", tag, len(student_trips))
    modes = ("classifier", "hmm") if do_hmm else ("classifier",)
    st, per_trip = evaluate_on_trips(student_trips, cfg, scorer, modes=modes)
    per_trip.to_csv(res_dir / "student_test_per_trip.csv", index=False)
    plots.confusion(st["pair_confusion_matrix"], res_dir / "confusion_matrix.png",
                    f"{tag}: student-test candidate pairs")
    _history_plots(tag, pipe, res_dir)
    importance = _importance(tag, pipe, res_dir)

    summary = {
        "model": tag, "base_model": spec.name, "family": spec.family,
        "key_config": json.dumps(params, default=str),
        "cv_macro_f1_mean": folds["macro_f1"].mean(), "cv_macro_f1_std": folds["macro_f1"].std(),
        "cv_top1_mean": folds["top1_accuracy"].mean(), "cv_top1_std": folds["top1_accuracy"].std(),
        "cv_log_loss_mean": folds["log_loss"].mean(), "cv_log_loss_std": folds["log_loss"].std(),
        "train_top1": train_m["top1_accuracy"], "train_macro_f1": train_m["macro_f1"],
        "train_log_loss": train_m["log_loss"],
        "student_top1": st["classifier_top1"], "student_route_continuity": st["classifier_continuity"],
        "student_seq_f1": st["classifier_seq_f1"],
        "student_hmm_top1": st.get("hmm_top1", np.nan),
        "student_hmm_route_continuity": st.get("hmm_continuity", np.nan),
        "student_hmm_seq_f1": st.get("hmm_seq_f1", np.nan),
        "student_macro_f1": st["pair_macro_f1"], "student_precision": st["pair_precision"],
        "student_recall": st["pair_recall"], "student_f1": st["pair_f1"],
        "student_accuracy": st["pair_accuracy"], "student_log_loss": st["pair_log_loss"],
        "student_roc_auc": st["pair_roc_auc"], "student_candidate_miss_rate": st["candidate_miss_rate"],
        "train_time_s": fit_s, "inference_ms_per_1k_pairs": st["inference_ms_per_1k_pairs"],
        "inference_ms_per_trip": st["inference_ms_per_trip"], "model_size_kb": _dir_size_kb(model_dir),
    }
    (res_dir / "metrics.json").write_text(json.dumps(
        {"summary": summary, "student_test": st, "train_subset": train_m,
         "cv_folds": folds.to_dict("records"), "feature_importance": importance}, indent=2, default=float))
    log.info("[%s] CV top1 %.4f±%.4f | student top1 %.4f (HMM %.4f) | macroF1 %.4f",
             tag, summary["cv_top1_mean"], summary["cv_top1_std"], summary["student_top1"],
             summary["student_hmm_top1"], summary["student_macro_f1"])
    return summary


def comparison_plots(df: pd.DataFrame, folds: pd.DataFrame, out: Path):
    plots.bar_metric(df, "cv_macro_f1_mean", out / "cv_macro_f1.png",
                     "Cross-validation Macro F1 (mean ± std over trip-grouped folds)", err="cv_macro_f1_std")
    plots.bar_metric(df, "cv_top1_mean", out / "cv_top1.png",
                     "Cross-validation observation Top-1 accuracy", err="cv_top1_std")
    plots.grouped_bars(df, ["student_top1", "student_macro_f1"], ["Top-1 accuracy", "Macro F1"],
                       out / "student_top1_macro_f1.png", "Student-testing: Top-1 accuracy and Macro F1")
    plots.bar_metric(df, "student_log_loss", out / "student_log_loss.png",
                     "Student-testing log loss (lower is better)", lower_is_better=True)
    plots.grouped_bars(df, ["student_top1", "student_hmm_top1"], ["classifier only", "classifier + HMM"],
                       out / "classifier_vs_hmm_top1.png", "Student-testing Top-1: classifier only vs + HMM")
    plots.grouped_bars(df, ["student_route_continuity", "student_hmm_route_continuity"],
                       ["classifier only", "classifier + HMM"], out / "route_continuity.png",
                       "Student-testing route continuity")
    plots.bar_metric(df, "inference_ms_per_1k_pairs", out / "inference_time.png",
                     "Inference time (ms per 1,000 candidate pairs)", lower_is_better=True)
    if len(folds):
        plots.cv_boxplot(folds, "top1_accuracy", out / "cv_top1_folds.png", "Top-1 accuracy across CV folds")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated subset of model names")
    ap.add_argument("--no-hmm", action="store_true", help="skip classifier+HMM evaluation per model")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("train", cfg["paths"]["results_dir"])
    specs = default_specs(cfg["training"]["n_jobs"])
    names = args.models.split(",") if args.models else list(specs)
    unknown = set(names) - set(specs)
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}; available: {sorted(specs)}")

    train = build_partition(cfg, "train", use_cache=not args.no_cache)
    log.info("Training pairs: %d rows, %d trips, positive rate %.3f", len(train.pairs),
             train.pairs["trip_id"].nunique(), train.pairs["label"].mean())
    student = load_labelled_trips(cfg, "student_test")
    if not student:
        raise SystemExit("No labelled student-testing trips found")

    results_dir = resolve(cfg["paths"]["results_dir"])
    rows = []
    for name in names:
        spec = specs[name]
        rows.append(run_model(spec, dict(spec.params), train, student, cfg, name, do_hmm=not args.no_hmm))

    df = pd.DataFrame(rows)
    comp_path = results_dir / "model_comparison.csv"
    if comp_path.exists() and args.models:  # merge with previous runs of other models
        prev = pd.read_csv(comp_path)
        df = pd.concat([prev[~prev["model"].isin(df["model"])], df], ignore_index=True)
    df.to_csv(comp_path, index=False)
    folds = pd.concat([pd.read_csv(p) for p in (results_dir / "models").glob("*/cv_folds.csv")
                       if p.parent.name in set(df["model"])], ignore_index=True)
    comparison_plots(df, folds, results_dir / "plots")

    primary = cfg["training"]["primary_metric"]
    ranked = df.sort_values(["cv_top1_mean", "student_top1"], ascending=False)
    selection = {
        "primary_metric": f"cross-validated observation-level {primary} (mean over trip-grouped folds)",
        "tie_breakers": ["student_top1", "student_log_loss", "inference_ms_per_1k_pairs"],
        "ranking": ranked[["model", "cv_top1_mean", "cv_top1_std", "student_top1", "student_hmm_top1",
                           "student_log_loss"]].to_dict("records"),
        "selected_model": ranked.iloc[0]["base_model"],
    }
    (results_dir / "model_selection.json").write_text(json.dumps(selection, indent=2, default=float))
    log.info("Selected for optimisation: %s", selection["selected_model"])


if __name__ == "__main__":
    main()
